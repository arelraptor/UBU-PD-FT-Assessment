from . import config

import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy import signal
from scipy.signal import stft

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def extract_all_features(dataset_identifier):
    """
    Extract all features of the incoming dataset, saving results to output_files dir

    Arg:
        dataset_identifier (string): String that identifies the dataset.

    Returns:
        None
    """    
    extract_tsfresh_features(dataset_identifier)
    extract_classical_features(dataset_identifier)
    extract_fi_tsfresh_features(dataset_identifier)

def extract_tsfresh_features(dataset_identifier):
    """
    Extract tsfresh features of the incoming dataset, saving results to output_files dir

    Arg:
        dataset_identifier (string): String that identifies the dataset.

    Returns:
        None
    """        
    #Read time series data and prepare it
    data_final=pd.read_csv(config.output_files_dir+dataset_identifier+"_data_time_series.csv",dtype={"ID": str})
    data_final = data_final.reset_index().sort_values(by=['ID', 'index']).set_index('index')
    data_final=data_final.drop(columns=['Unnamed: 0'])
    data_final['FREQUENCY'] = data_final['FREQUENCY'].fillna(0)
    
    #Read diagnostic csv containing UPDRS ratings
    y_diagnostic=pd.read_csv(config.input_files_dir+dataset_identifier+"_diagnostic.csv",dtype={'UPDRS': np.int32,"ID": str})
    y_diagnostic=y_diagnostic.sort_values(by=['ID'])

    #Filter data to ensure that we have some ID in both sides due to probable video rejections
    list_id_y_diagnostic=np.unique(np.array(y_diagnostic["ID"]))
    list_id_data_final=np.unique(np.array(data_final["ID"]))
    data_final=data_final[data_final.ID.isin(list_id_y_diagnostic)]
    y_diagnostic=y_diagnostic[y_diagnostic.ID.isin(list_id_data_final)]
    
    #Building our diagnostics Series
    y_all=pd.Series(data=y_diagnostic["UPDRS"].to_numpy(),index=y_diagnostic["ID"].to_numpy())
    
    #TSFresh invocation for capturing all feautes
    extracted_features = extract_features(data_final, column_id="ID", impute_function=impute)
    selected_features = select_features(extracted_features, y_all, multiclass=True, n_significant=3,ml_task="classification")
    
    selected_features.to_csv(config.output_files_dir+dataset_identifier+"_tsfresh_features.csv")
    

def extract_classical_features(dataset_identifier):
    """
    Extract classical features of the incoming dataset, saving results to output_files dir

    Arg:
        dataset_identifier (string): String that identifies the dataset.

    Returns:
        None
    """        
    #Read time series data and prepare it
    data_final=pd.read_csv(config.output_files_dir+dataset_identifier+"_data_time_series.csv",dtype={"ID": str})
    data_final = data_final.reset_index().sort_values(by=['ID', 'index']).set_index('index')
    data_final=data_final.drop(columns=['Unnamed: 0'])
    
    #Read Dataframe that contains fps for each video
    fps_per_video=pd.read_csv(config.output_files_dir+dataset_identifier+"_fps_videos.csv",dtype={"ID": str,"FPS":float})
    fps_per_video=fps_per_video.set_index("ID")
    
    #Read diagnostic csv containing UPDRS ratings
    y_diagnostic=pd.read_csv(config.input_files_dir+dataset_identifier+"_diagnostic.csv",dtype={'UPDRS': np.int32,"ID": str})
    y_diagnostic=y_diagnostic.sort_values(by=['ID'])

    #Filter data to ensure that we have some ID in both sides due to probable video rejections
    list_id_y_diagnostic=np.unique(np.array(y_diagnostic["ID"]))
    list_id_data_final=np.unique(np.array(data_final["ID"]))
    data_final=data_final[data_final.ID.isin(list_id_y_diagnostic)]
    y_diagnostic=y_diagnostic[y_diagnostic.ID.isin(list_id_data_final)]
    
    summary_df=data_final.groupby('ID').agg({
        'SMOOTHED_AMPLITUDE': ['mean', 'std', 'max'],
        'VELOCITY': ['mean', 'max', 'std'],
        'ACCELERATION': ['mean', 'max', 'std'],
        'DISTANCE_ANG': ['mean', 'std'],
        'FREQUENCY': ['mean', 'std', 'count']
    })
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    
    # Video FPS

    # Array for savidng the results by video
    regularity = []

    # Group by video
    for video_id, group in data_final.groupby('ID'):
        amplitude = group['SMOOTHED_AMPLITUDE'].values
        
        fps = fps_per_video.at[video_id, "FPS"]
        
        # Detect peaks with minimum separation (e.g. 0.3s)
        peaks, _ = find_peaks(amplitude, distance=fps//3)

        # Calcular tiempos de los picos
        peak_times = peaks * (1 / fps)

        # Calculate peak times
        periods = np.diff(peak_times)

        # Calculate regularity metrics
        if len(periods) >= 2:
            interval_std = np.std(periods)
            interval_cv = interval_std / np.mean(periods) if np.mean(periods) != 0 else np.nan
        else:
            interval_std = np.nan
            interval_cv = np.nan

        regularity.append({
            'ID': video_id,
            'INTERVAL_STD': interval_std,
            'INTERVAL_CV': interval_cv
        })

    # Convert to DataFrame
    regularity_df = pd.DataFrame(regularity)

    # Merge to the original summary_df
    summary_df = summary_df.merge(regularity_df, on='ID', how='left')

    # Add amplitude slope per video
    summary_df['AMPLITUDE_SLOPE'] = data_final.groupby('ID').apply(calculate_slope_amplitude).values
    
    # Add frequency variation per video
    summary_df['FREQUENCY_DROP'] = data_final.groupby('ID').apply(calculate_average_frequency).values
    
    # Add acceleration variation per video
    summary_df['ACCELERATION_INCREASE'] = data_final.groupby('ID').apply(calculate_average_acceleration).values
    
    # Apply the function to each group of 'id' and make sure the output is a DataFrame
    summary_stats = data_final.groupby('ID').apply(calculate_skew_kurt)
    # Rename the result columns to match the columns we want to add
    summary_stats.columns = ['SKEW_AMPLITUDE', 'KURTOSIS_AMPLITUDE', 'SKEW_VELOCITY', 'KURTOSIS_VELOCITY']
    # Make sure summary_stats has only the metric values ​​per video (id)
    summary_stats = summary_stats.reset_index()
    # Now we add these new columns to summary_df based on 'id'
    summary_df = summary_df.merge(summary_stats, on='ID', how='left')
    
    # We apply the function to obtain FFT features
    fft_features = data_final.groupby('ID').apply(calculate_fft_features)
    # Renombramos las columnas
    fft_features.columns = ['FFT_DOMINANT_FREQ_AMPLITUDE', 'FFT_POWER_AMPLITUDE', 'FFT_DOMINANT_FREQ_VELOCITY', 'FFT_POWER_VELOCITY']
    # We added new features to summary_df based on 'id'
    summary_df = summary_df.merge(fft_features, on='ID', how='left')

    summary_df.set_index('ID', inplace=True)
    summary_df.index.name = None
    
    summary_df.to_csv(config.output_files_dir+dataset_identifier+"_classical_features.csv")

def extract_fi_tsfresh_features(dataset_identifier):
    """
    Extract FI+tsfresh features of the incoming dataset, saving results to output_files dir

    Arg:
        dataset_identifier (string): String that identifies the dataset.

    Returns:
        None
    """        
    #Read time series data and prepare it
    data_final=pd.read_csv(config.output_files_dir+dataset_identifier+"_data_time_series.csv",dtype={"ID": str})
    data_final = data_final.reset_index().sort_values(by=['ID', 'index']).set_index('index')
    data_final=data_final.drop(columns=['Unnamed: 0'])
    data_final['FREQUENCY'] = data_final['FREQUENCY'].fillna(0)
    
    #Read Dataframe that contains fps for each video
    fps_per_video=pd.read_csv(config.output_files_dir+dataset_identifier+"_fps_videos.csv",dtype={"ID": str,"FPS":float})
    fps_per_video=fps_per_video.set_index("ID")
    
    #Read diagnostic csv containing UPDRS ratings
    y_diagnostic=pd.read_csv(config.input_files_dir+dataset_identifier+"_diagnostic.csv",dtype={'UPDRS': np.int32,"ID": str})
    y_diagnostic=y_diagnostic.sort_values(by=['ID'])

    #Filter data to ensure that we have some ID in both sides due to probable video rejections
    list_id_y_diagnostic=np.unique(np.array(y_diagnostic["ID"]))
    list_id_data_final=np.unique(np.array(data_final["ID"]))
    data_final=data_final[data_final.ID.isin(list_id_y_diagnostic)]
    y_diagnostic=y_diagnostic[y_diagnostic.ID.isin(list_id_data_final)]
    
    #Building our diagnostics Series
    y_all=pd.Series(data=y_diagnostic["UPDRS"].to_numpy(),index=y_diagnostic["ID"].to_numpy())
    
    result_rows = []

    for id_val, group in data_final.groupby('ID'):
        amplitude = group['SMOOTHED_AMPLITUDE'].values
        
        fps= fps_per_video.at[id_val, "FPS"]
        
        t, f, Zxx, max_freq, max_intensity = peakFreqInte_bySTFT(amplitude, fs=fps, nperseg=100, noverlap=None,
            f_lower_cutoff=0.5, f_upper_cutoff=10)
    
        # Create a DataFrame for this id
        df_temp = pd.DataFrame({
            'ID': id_val,
            't': t,
            'max_freq': max_freq,
            'max_intensity': max_intensity
        })
    
        result_rows.append(df_temp)

    # Concatenate all results
    series_df = pd.concat(result_rows, ignore_index=True)
    series_df['IF_value'] = series_df['max_freq'] * series_df['max_intensity']
    series_df=series_df.drop(columns=['t'])
    
    extracted_features = extract_features(series_df, column_id="ID", impute_function=impute)
    selected_features = select_features(extracted_features, y_all, multiclass=True, n_significant=3,ml_task="classification")
    selected_features.to_csv(config.output_files_dir+dataset_identifier+"_fi_tsfresh_features.csv")
    

def calculate_slope_amplitude(group):
    """
    Calculate the amplitude slope across video frames

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video

    Returns:
        float: Slope of the fitted line
    """        
    x = np.arange(len(group))
    y = group['SMOOTHED_AMPLITUDE'].values
    #Slope of the fitted line
    slope = np.polyfit(x, y, 1)[0] 
    return slope

def calculate_average_frequency(group):
    """
    Calculate the difference in mean frequency between the first and second half of a video.

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video

    Returns:
        float: The difference in mean frequency between the first and second half of a video. If there is a decrease, it will be negative
    """     
    # Average frequency in the first and last part of the video
    first_half = group.head(len(group)//2)['FREQUENCY'].mean()
    second_half = group.tail(len(group)//2)['FREQUENCY'].mean()
    return first_half - second_half   

def calculate_average_acceleration(group):
    """
    Calculate the difference in mean acceleration between the first and second half of a video.

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video

    Returns:
        float: The difference in mean acceleration between the first and second half of a video. If there is a decrease, it will be negative
    """        
    first_half = group.head(len(group)//2)['ACCELERATION'].mean()
    second_half = group.tail(len(group)//2)['ACCELERATION'].mean()
    return first_half - second_half  # Si hay un aumento, será positivo


def calculate_skew_kurt(group):
    """
    Calculate the skewness and kurtosis for amplitude and velocity columns .

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video

    Returns:
        pandas.Series: A series containing four values in the following order:
            [0] skew_amplitude (float): Skewness of 'SMOOTHED_AMPLITUDE'.
            [1] kurtosis_amplitude (float): Kurtosis of 'SMOOTHED_AMPLITUDE'.
            [2] skew_velocity (float): Skewness of 'VELOCITY'.
            [3] kurtosis_velocity (float): Kurtosis of 'VELOCITY'.
    """   
    skew_amplitude = skew(group['SMOOTHED_AMPLITUDE'])
    kurtosis_amplitude = kurtosis(group['SMOOTHED_AMPLITUDE'])
    
    skew_velocity = skew(group['VELOCITY'])
    kurtosis_velocity = kurtosis(group['VELOCITY'])

    return pd.Series([skew_amplitude, kurtosis_amplitude, skew_velocity, kurtosis_velocity])

def calculate_fft(group, column):
    """
    Calculate the Fast Fourier Transform (FFT) of a signal column.

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video
        column (str): Name of the column containing the signal to analyze

    Returns:
        pandas.Series: A series containing two values:
            [0] dominant_freq (float): The dominant frequency component of the signal (in normalized units).
            [1] power_spectrum (float): The total spectral power (sum of squared magnitudes of the FFT).
    """     
    # We calculate the FFT of the signal in 'column'
    signal = group[column].values
    n = len(signal)
    
    # Apply FFT
    fft_result = np.fft.fft(signal)
    
    # Calculate corresponding frequencies
    freqs = np.fft.fftfreq(n)
    
    # Take only the positive half of the spectrum (since the FFT is symmetrical)
    fft_result = fft_result[:n//2]
    freqs = freqs[:n//2]
    
    # Calculate the magnitude of the FFT (in absolute value)
    fft_magnitude = np.abs(fft_result)
    
    # Find the dominant frequency (the one with the greatest magnitude)
    dominant_freq = freqs[np.argmax(fft_magnitude)]
    
    # Calculate the spectral power (sum of the squared magnitudes)
    power_spectrum = np.sum(fft_magnitude**2)
    
    return pd.Series([dominant_freq, power_spectrum])


def calculate_fft_features(group):
    """
    Compute FFT-based features (dominant frequency and spectral power) for amplitude and velocity signals

    Arg:
        group (Panda Dataframe): Panda dataframe that contains all frames belonging to a single video

    Returns:
        pandas.Series: A series containing four values in the following order:
            [0] amplitude_dominant_freq (float): Dominant frequency of the amplitude signal.
            [1] amplitude_power (float): Total spectral power of the amplitude signal.
            [2] velocity_dominant_freq (float): Dominant frequency of the velocity signal.
            [3] velocity_power (float): Total spectral power of the velocity signal.
    """      
    # FFT para AMPLITUDE
    amplitude_dominant_freq, amplitude_power = calculate_fft(group, 'SMOOTHED_AMPLITUDE')
    
    # FFT para VELOCITY
    velocity_dominant_freq, velocity_power = calculate_fft(group, 'VELOCITY')
    
    return pd.Series([amplitude_dominant_freq, amplitude_power, velocity_dominant_freq, velocity_power])

def peakFreqInte_bySTFT(sig, fs, nperseg=150, noverlap=100, f_lower_cutoff=1, f_upper_cutoff=10):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal and extract the peak frequency and intensity within a given frequency range for each time segment.

    Args:
        sig (array-like): 
            Input signal (1D array).
        fs (float): 
            Sampling frequency of the signal (in Hz).
        nperseg (int, optional): 
            Length of each segment for STFT computation. Default is 150.
        noverlap (int, optional): 
            Number of points to overlap between segments. Default is 100.
        f_lower_cutoff (float, optional): 
            Lower frequency cutoff for analysis (in Hz). Default is 1.
        f_upper_cutoff (float, optional): 
            Upper frequency cutoff for analysis (in Hz). Default is 10.

    Returns:
        tuple:
            t (numpy.ndarray): 
                Time vector of the STFT segments.
            f (numpy.ndarray): 
                Frequency bins corresponding to the rows of `Zxx`.
            Zxx (numpy.ndarray): 
                Magnitude of the STFT result (2D array of shape [frequencies, times]).
            max_freq (list of float): 
                Peak frequency (Hz) within the specified range for each time segment.
            max_intensity (numpy.ndarray): 
                Maximum intensity (amplitude) at the peak frequency for each time segment.
    """    
    f, t, Zxx = signal.stft(sig, fs, nperseg=nperseg, noverlap=noverlap)
    Zxx = np.abs(Zxx)

    f_min_arg = np.argwhere(f > f_lower_cutoff)[0][0]
    f_max_arg = np.argwhere(f < f_upper_cutoff)[-1][0]
    f_range = f[f_min_arg:f_max_arg]

    max_intensity = Zxx[f_min_arg:f_max_arg].max(axis=0)
    max_freq = []
    for idx, _ in enumerate(max_intensity):
        max_freq.append(f_range[np.argmax(Zxx[f_min_arg:f_max_arg, idx])])

    return t, f, Zxx, max_freq, max_intensity
