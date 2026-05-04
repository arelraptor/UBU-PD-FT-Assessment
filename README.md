# Artificial Intelligence for Parkinson's Disease Evaluation: an experimental study on finger tapping

## Description

This repository is the official implementation of: (future link to the paper)

<img src="data/StepsFTinCV.png" width="900">

Here, you can find the code for performing an experimental study using finger tapping videos. In the next sections, you can find the steps to configure and use the code that is present in this repository.

## Dataset

Dataset for FIS videos is available at https://zenodo.org/records/17738775. 

This dataset contains 234 video recordings of controls and patients (individuals diagnosed with Parkinson’s disease) performing the standardized finger tapping test, a commonly used motor assessment in clinical and research settings. The videos were collected as part of a collaborative study conducted by researchers from the University of Burgos and the Hospital Universitario de Burgos. This work was supported by the project PI19/00670 of the Ministerio de Ciencia, Innovación y Universidades, Instituto de Salud Carlos III, Spain.

The dataset is intended to support research on motor symptom characterization, quantitative assessment methods, and the development of automated analysis tools for Parkinson’s disease. All recordings were obtained following appropriate ethical approvals, and participants provided informed consent for research use of their data.

All the participants and the Parkinson's Disease Association are thanked for their support. Similarly, we thank Dr. Gamez Leiva and Dr. Madrigal for the videos assessments.

The structure of the zip file is:

- fis_diagnostic.csv. Table with two columns, the first one is the name of the video and the second one the UPDRS rating for the clip.
- videos. Folder with 234 videos (left and right hands).

## Classification Workflow

The following diagram illustrates the nested cross-validation pipeline used in this study to ensure robust model evaluation and hyperparameter tuning.

![Methodology Workflow](data/fullProcess_v2.png)

### 1. Input Data
The process begins with the original **Dataset ($N$ samples)**. Due to the nature of the data, a rigorous validation strategy is employed to prevent overfitting and provide an unbiased estimate of model performance.

### 2. Outer Loop: Leave-One-Out Cross-Validation
To maximize the use of available data, an **Outer Loop** utilizing **Leave-one-out CV** is implemented:
* **Train fold:** $N-1$ samples are used for training and hyperparameter optimization.
* **Test fold:** 1 sample is held out for independent evaluation.
* **Repetition:** This cycle is repeated $N$ times, ensuring every sample in the dataset is used as a test set exactly once.

### 3. Classifiers & Inner Loop (Hyperparameter Tuning)
For each iteration of the outer loop, the training data is passed to six different classification algorithms:
* **Linear Regression**
* **k-Nearest Neighbors (kNN)**
* **Decision Tree**
* **Random Forest**
* **AdaBoost**
* **XGBoost**.

Within the **Inner Loop**, we perform **GridSearchCV** to identify the best parameters for each model. The optimization process uses the **Matthews Correlation Coefficient (MCC)** as the primary scoring metric to be maximized, ensuring balanced performance across all classes.

### 4. Output and Evaluation Metrics
Once the optimal models are determined and tested against the held-out samples, the final performance is aggregated. The pipeline outputs a comprehensive set of evaluation metrics:
* **MCC** 
* **F1-Score**
* **Accuracy**
* **A-AC** (Acceptable Accuracy)
* **Precision**
* **Recall**

## Code Information

This code has been written in Python and a Jupyter Notebook is also used for running the pipeline. In te sections below, you can find the requirements and how to use this code.

## Requirements

In this project you can find two different files for recreating the conda environment:

a) To install only packages, run:

```setup
conda create --name new_env --file requirements.txt
```

b) To import full environment, run:

```setup
conda env create -f environment.yml
```

## Usage Instructions 

```notebooks\Main.ipynb``` provides step-by-step instructions for processing data.

a) First of all, you have to add your UPDRS ratings file to ```input_files``` folder. Name of this file must be ```<dataset_identifier>_diagnostic.csv```. This ```<dataset_identifier>``` must be the same as you will use inside main notebook ```notebooks\Main.ipynb```

This file must contain two columns, called *ID* and *UPDRS*, separated by comma "*,*".
- *ID*: name of each video file, without extension
- *UPDRS*: UPDRS rating per video
```csv  
ID,UPDRS
video1,0
video2,3
...
```


b) Please, review ```lib\config.py```. Current variables fit the folder structure that it is shown under this project.

c) Using MediaPipe, ```process_video``` function will generate the baseline kinematic measures time series. As result, these files are generated: 

- ```output_files\<dataset_identifier>_time_series.csv```: baseline kinematic measures time series.
- ```output_files\<dataset_identifier>_fps_videos.csv```: containing the fps for each video.
- ```log\<dataset_identifier>_frame_rate_processed_videos.csv```: for each successfully processed video, the percentage of frames where the hand is detected successfully is shown.
- ```log\<dataset_identifier>_frame_rate_rejected_videos.csv```: for each rejected video, the percentage of frames where the hand is detected successfully is shown. These videos are excluded for next steps.

d) Function ```extract_all_features``` extracts all the features using all approaches (Classical, TsFresh and FI+TsFresh). As output, these files are generated:

- ```output_files\<dataset_identifier>_classical_features.csv```.
- ```output_files\<dataset_identifier>_tsfresh_features.csv```.
- ```output_files\<dataset_identifier>_fi_tsfresh_features.csv```.

e) Finally, function ```classify_video``` performs machine learning models training and validation. As input, ```features_type``` value must be provided. Accepted values are: *classical*, *tsfresh* and *fi_tsfresh*. As result, these files are generated: 

- ```log\<dataset_identifier>_<features_type>_execution.csv```: tracks the progress of the execution.
- ```result_files\<dataset_identifier>_<features_type>_result.csv```: shows a detailed report for each iteration.
- ```result_files\<dataset_identifier>_<features_type>_execution_summary.csv```: this files provides a summary of each algorithm’s performance and the global outcome for each evaluated metric.
- ```result_files\confusion_matrix\<dataset_identifier>_<features_type>_cm_<ml_algorithm>.png```: final confusion matrix for each evaluated algorithm.
- ```result_files\roc_curves\<dataset_identifier>_<features_type>_roc_<ml_algorithm>.png```: roc curves for each evaluated algorithm.

## References

If you use this code in your research, please cite our paper

```
Pending
```
