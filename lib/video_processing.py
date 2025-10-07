from . import config

import os
import cv2
import math
import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from tqdm.notebook import tqdm


from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def process_video(path_videos, dataset_identifier):
    """
    Process videos to extract hand movement features using MediaPipe HandLandmarker.

    This function:
    - Iterates through all video files in the given directory.
    - Detects hand landmarks (index finger, thumb, wrist) frame by frame.
    - Discards frames without a detected hand.
    - Calculates angles between wrist-index and wrist-thumb vectors.
    - Computes hand size, thumb-index amplitude, and normalized amplitude.
    - Smooths the normalized amplitude and calculates its velocity and acceleration.
    - Detects peaks in the smoothed amplitude to estimate instantaneous frequency.
    - Saves extracted features into a DataFrame per video, including:
        DISTANCE_ANG, SMOOTHED_AMPLITUDE, VELOCITY, ACCELERATION, FREQUENCY.
    - Calculates and saves the percentage of usable frames per video.
    - Filters out videos with less than 90% good frames.
    - Saves CSV files for:
        - Processed time series data for all videos
        - FPS of each video
        - Percentage of good frames
        - Rejected videos due to low frame quality

    Args:
        path_videos (str): Path to the folder containing video files.
        dataset_identifier (str): Identifier for naming output files.

    Returns:
        None: Results are saved to CSV files in the directories specified by `config`.
    """    
    #Columns index finger
    cols_index=['INDEX_X','INDEX_Y','INDEX_Z']
    #Columns thumb
    cols_thumb=['THUMB_X','THUMB_Y','THUMB_Z']
    #columnas wrist
    cols_wrist=['WRIST_X','WRIST_Y','WRIST_Z']
    
    #Variables for output files
    data_final=pd.DataFrame()
    frame_percentage_videos=pd.DataFrame()
    frame_ratio_videos=pd.DataFrame()
    frame_percentage_rejected_videos=pd.DataFrame()
    
    with os.scandir(path_videos) as files:
        files = sorted(files, key=lambda entry: entry.name)
        files = [file.name for file in files if file.is_file()]

    #Analyzing all the videos    
    for file in tqdm(files, desc="Processing videos", unit="video"):
        #Intermediate variables to discard noisy videos
        good_frames_count=0
        bad_frames_count=0
        all_frames_count=0
        good_frames_percentage=0
        
        #Extract file name without extension
        id_user=file.split('.')[0]
        
        #Initialize MediaPipe options
        base_options = BaseOptions(model_asset_path='../utils/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE, num_hands=1)
        detector = HandLandmarker.create_from_options(options)
        
        frames=[]
        hands=[]
        handedness=None
        
        cap = cv2.VideoCapture(path_videos+file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame using HandLandmarker
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)
            #Check hand detection
            if results.handedness:
                hands.append(results.hand_landmarks)
                hand_video=results.handedness[0][0].category_name
                good_frames_count=good_frames_count+1
            else:
                bad_frames_count=bad_frames_count+1

        cap.release()

        all_frames_count=good_frames_count+bad_frames_count
        good_frames_percentage=(good_frames_count/(all_frames_count))*100
        
        frame_ratio_videos=pd.concat([frame_ratio_videos,pd.DataFrame(
                                                {'ID': id_user,
                                                  'FPS' : [fps]
                                                }
                                                )], ignore_index=True)
        
        if (good_frames_percentage > 90):
            df = pd.DataFrame()
            previous_distance=0
            previous_speed=0
            for hand in hands:
                for hand_landmarks in hand:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

                    temp_df= pd.DataFrame({'INDEX_X': [hand_landmarks_proto.landmark[8].x ],
                                           'INDEX_Y': [hand_landmarks_proto.landmark[8].y ],
                                           'INDEX_Z': [hand_landmarks_proto.landmark[8].z ],
                                           'THUMB_X': [hand_landmarks_proto.landmark[4].x ],
                                           'THUMB_Y': [hand_landmarks_proto.landmark[4].y ],
                                           'THUMB_Z': [hand_landmarks_proto.landmark[4].z ],
                                           'WRIST_X': [hand_landmarks_proto.landmark[0].x ],
                                           'WRIST_Y': [hand_landmarks_proto.landmark[0].y ],
                                           'WRIST_Z': [hand_landmarks_proto.landmark[0].z ]
                            })
                    
                    #Calculate the angle for each wrist-index vs. wrist-thumb combination
                    temp_df["angle"]=angle_of_vectors(temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_index].values,temp_df.iloc[0][cols_wrist].values - temp_df.iloc[0][cols_thumb].values)
                    #Concat temp datagrame to the intermediate one
                    df=pd.concat([df,temp_df], ignore_index=True)

            
            df["DISTANCE_ANG"]=df["angle"]/90
            
            df['HAND_SIZE'] = df.apply(calculate_hand_size, axis=1)

            # Calculate the amplitude (distance between the thumb and the index finger)
            df['AMPLITUDE'] = df.apply(calculate_amplitude, axis=1)

            # Normalize the amplitude using the distance between the wrist and the index finger
            df['NORMALIZED_AMPLITUDE'] = (df['AMPLITUDE'] / df['HAND_SIZE']) 

            # Smoothing of the normalized amplitude
            df['SMOOTHED_AMPLITUDE'] = df['NORMALIZED_AMPLITUDE'].rolling(window=3, center=True).mean()
            df['SMOOTHED_AMPLITUDE'] = df['SMOOTHED_AMPLITUDE'].fillna(method='bfill').fillna(method='ffill')

            
            # Define the frames per second (FPS) of the video
            delta_t = 1 / fps  # Time between frames in seconds

            # Velocity (first derivative)
            df['VELOCITY'] = df['SMOOTHED_AMPLITUDE'].diff() / delta_t
            df['VELOCITY'] = df['VELOCITY'].fillna(0)

            # Acceleration (second derivative)
            df['ACCELERATION'] = df['VELOCITY'].diff() / delta_t
            df['ACCELERATION'] = df['ACCELERATION'].fillna(0)
            
            
            peaks, _ = find_peaks(df['SMOOTHED_AMPLITUDE'], distance=fps//3)  # minimum distance between peaks

            # Calculate the time of each peak
            peak_times = peaks * (1 / fps)  # Convert frame index to time in seconds
            # Calculate time differences between peaks
            periods = np.diff(peak_times)  # in seconds
            # Instantaneous frequency per pair of peaks (Hz)
            frequencies = 1 / periods

            # Adding to the DataFrame as a column (aligned with the second peak of each pair)
            df['FREQUENCY'] = np.nan
            df.loc[peaks[1:], 'FREQUENCY'] = frequencies
          
            
            temp_df_final=pd.DataFrame(df[["DISTANCE_ANG","SMOOTHED_AMPLITUDE","VELOCITY","ACCELERATION","FREQUENCY"]])
            temp_df_final["ID"]=id_user

            data_final=pd.concat([data_final,temp_df_final], ignore_index=True)

            #Build the series with the final output
            frame_percentage_videos=pd.concat([frame_percentage_videos,pd.DataFrame(
                                                {'ID': id_user,
                                                  'Percentage' : [good_frames_percentage]
                                                }
                                                )], ignore_index=True)
        else:
            frame_percentage_rejected_videos=pd.concat([frame_percentage_rejected_videos,pd.DataFrame(
                                                {'ID': id_user,
                                                  'Porcentaje' : [good_frames_percentage]
                                                }
                                                )], ignore_index=True)
    
    frame_percentage_videos = frame_percentage_videos.set_index("ID")
    frame_percentage_videos.to_csv(config.log_files_dir+dataset_identifier+"_frame_rate_processed_videos.csv")
    
    frame_percentage_rejected_videos = frame_percentage_rejected_videos.set_index("ID")
    frame_percentage_rejected_videos.to_csv(config.log_files_dir+dataset_identifier+"_frame_rate_rejected_videos.csv")
    
    frame_ratio_videos = frame_ratio_videos.set_index("ID")
    frame_ratio_videos.to_csv(config.output_files_dir+dataset_identifier+"_fps_videos.csv")
    
        
    data_final.to_csv(config.output_files_dir+dataset_identifier+"_data_time_series.csv")

    

def euclidean_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Euclidean distance between two 3D points.

    Args:
        x1, y1, z1 (float): Coordinates of the first point.
        x2, y2, z2 (float): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """    
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)    
    

def angle_of_vectors (vector_1,vector_2):
    """
    Calculate the angle in degrees between two vectors in n-dimensional space.

    Args:
        vector_1 (np.array): First vector.
        vector_2 (np.array): Second vector.

    Returns:
        float: Angle between the two vectors in degrees.
    """
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

def calculate_hand_size(row):
    """
    Calculate the approximate size of the hand as the Euclidean distance between the wrist and the tip of the index finger.

    Args:
        row (pd.Series): A row from a DataFrame containing the following columns:
                         'WRIST_X', 'WRIST_Y', 'WRIST_Z', 'INDEX_X', 'INDEX_Y', 'INDEX_Z'.

    Returns:
        float: Euclidean distance between the wrist and the index finger tip.
    """
    wrist_index_distance = euclidean_distance(row['WRIST_X'], row['WRIST_Y'], row['WRIST_Z'],
                                              row['INDEX_X'], row['INDEX_Y'], row['INDEX_Z'])
    return wrist_index_distance


def calculate_amplitude(row):
    """
    Calculate the amplitude of finger movement as the Euclidean distance between the thumb tip and the index finger tip for a given frame.

    Args:
        row (pd.Series): A row from a DataFrame containing the following columns:
                         'THUMB_X', 'THUMB_Y', 'THUMB_Z', 
                         'INDEX_X', 'INDEX_Y', 'INDEX_Z'.

    Returns:
        float: Euclidean distance between the thumb tip and the index finger tip.
    """
    thumb_index_distance = euclidean_distance(row['THUMB_X'], row['THUMB_Y'], row['THUMB_Z'],
                                              row['INDEX_X'], row['INDEX_Y'], row['INDEX_Z'])
    return thumb_index_distance
    