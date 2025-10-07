import pandas as pd
import numpy as np

from . import config
from . import ml_models as m2t

from matplotlib import pyplot as plt

from sklearn.model_selection import LeaveOneOut,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef,roc_auc_score, confusion_matrix,ConfusionMatrixDisplay,cohen_kappa_score

from datetime import datetime

def classify_video(dataset_identifier,features_type):
    """
    Perform video classification using multiple machine learning models with Leave-One-Out cross-validation, and generate evaluation metrics 
    and visualizations.

    This function:
    - Reads diagnostic labels (UPDRS) for the specified dataset.
    - Reads extracted features corresponding to the dataset and feature type.
    - Filters the data to include only videos present in both diagnostics and features.
    - Trains and evaluates multiple models (from `m2t.models`) using Leave-One-Out CV.
    - Performs hyperparameter tuning with GridSearchCV for each model.
    - Calculates metrics including Accuracy, Acceptable Accuracy (prediction off by ≤1), Kappa, F1 score, MCC, and ROC AUC.
    - Saves individual predictions and results for each model.
    - Generates and saves aggregated confusion matrices for each model.
    - Saves a summary CSV of all evaluation metrics for all models.

    Args:
        dataset_identifier (str): Name or identifier of the dataset to process.
        features_type (str): Type of features to use for classification ("classical", "tsfresh" or "fi_tsfresh").

    Returns:
        None: Results are saved to CSV files and confusion matrices are saved as PNGs specified by `config`.
    """
    
    
    final_result=pd.DataFrame()
    #Variable for building confusion matrix
    conf_matrix_data = {m: {'y_true': [], 'y_pred': [], 'y_proba': []} for m in m2t.models.keys()}
    
    #Read diagnostic csv containing UPDRS ratings
    y_diagnostic=pd.read_csv(config.input_files_dir+dataset_identifier+"_diagnostic.csv",dtype={'UPDRS': np.int32,"ID": str})
    y_diagnostic=y_diagnostic.sort_values(by=['ID'])
    y_diagnostic=y_diagnostic.drop_duplicates()
    y_diagnostic=y_diagnostic.reset_index()
    y_diagnostic=y_diagnostic.drop(columns=['index'])
    
    #Read features data and prepare it
    features=pd.read_csv(config.output_files_dir+dataset_identifier+"_"+features_type+"_features.csv",dtype={"Unnamed: 0": str})
    features=features.rename(columns={"Unnamed: 0": "ID"})
    features=features.sort_values(by=['ID'])
    
    #Filter data to ensure that we have some ID in both sides due to probable video rejections
    list_id_y_diagnostic=np.unique(np.array(y_diagnostic["ID"]))
    list_id_data_final=np.unique(np.array(features["ID"]))
    features=features[features.ID.isin(list_id_y_diagnostic)]
    y_diagnostic=y_diagnostic[y_diagnostic.ID.isin(list_id_data_final)]    
    
    #Build numpy array from features
    features=features.drop(columns=['ID'])
    final_x_array=np.array(features)
    
    #Build numpy array from diagnotics
    y_diagnostic_array1d=np.array(y_diagnostic["UPDRS"])
    
    loo = LeaveOneOut()
    
    for train_loo, test_loo in loo.split(final_x_array):
        for model_name, model in m2t.models.items():
            #Record execution starts
            now = datetime.now()
            execution_file = open(config.log_files_dir+dataset_identifier+"_"+features_type+"_execution.txt", "a")
            execution_file.write(str(now)+"\t"+ model_name+"\t")
            execution_file.close()
            
            model_param_grid = m2t.model_parameter_rules[model]
            
            gridS = GridSearchCV(
                    estimator=model(),
                    param_grid=model_param_grid,
                    n_jobs=-1,
                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=config.my_random_state),
                    scoring=config.grid_search_scoring_metric
                )
            
            gridS.fit(final_x_array[train_loo], y_diagnostic_array1d[train_loo])
            
            #Calculate A-AC
            substraction=abs(y_diagnostic_array1d[test_loo] - gridS.predict(final_x_array[test_loo]))
            total_hits=np.count_nonzero((substraction == 1) | (substraction == 0))
            length_array_prediction=substraction.size
            acceptable_accurary_pct=total_hits/length_array_prediction
            
            data = {'Model': [model_name],
                    'Variables': [str(gridS.best_estimator_).replace("\n", "")], 
                    'Percentage_predict': [accuracy_score(y_diagnostic_array1d[test_loo], gridS.predict(final_x_array[test_loo]))],
                    'Percentage_acceptable_predict': [acceptable_accurary_pct],
                    'Kappa_score': [cohen_kappa_score(y_diagnostic_array1d[test_loo], gridS.predict(final_x_array[test_loo]))],
                    'F1 score': [f1_score(y_diagnostic_array1d[test_loo], gridS.predict(final_x_array[test_loo]),average='weighted')],
                    'MCC': [matthews_corrcoef(y_diagnostic_array1d[test_loo], gridS.predict(final_x_array[test_loo]))],
                    'value_real': [y_diagnostic_array1d[test_loo][0]],
                    'value_predict': [gridS.predict(final_x_array[test_loo])[0]]
                    }
            
            final_result=pd.concat([final_result,pd.DataFrame(data)], ignore_index=True)
            
            now2 = datetime.now()
            execution_file = open(config.log_files_dir+dataset_identifier+"_"+features_type+"_execution.txt", "a") 
            execution_file.write(str(now2-now)+"\n")
            execution_file.close()
            
            y_pred = gridS.predict(final_x_array[test_loo])
            y_true = y_diagnostic_array1d[test_loo]

            # Save predictions and actual labels for this model
            conf_matrix_data[model_name]['y_true'].extend(y_true)
            conf_matrix_data[model_name]['y_pred'].extend(y_pred)
            
            # Save probabilities (for ROC AUC)
            if hasattr(gridS.best_estimator_, "predict_proba"):
                y_proba = gridS.predict_proba(final_x_array[test_loo])
                conf_matrix_data[model_name]['y_proba'].extend(y_proba)

    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data['y_true'])
        y_pred_total = np.array(data['y_pred'])

        cm_total = confusion_matrix(y_true_total, y_pred_total)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_total)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{dataset_identifier}_{features_type}_{model_name} Agg", fontsize=14)
        # Save figure
        final_cm_filename = f"{config.results_files_dir}confusion_matrix/{dataset_identifier}_{features_type}_cm_{model_name}.png"
        plt.savefig(final_cm_filename)
        plt.close()

        
    summary_results = []
    
    for model_name, data in conf_matrix_data.items():
        y_true_total = np.array(data['y_true'])
        y_pred_total = np.array(data['y_pred'])
        y_proba_total = np.array(data['y_proba'])

        # Classical metrix
        acc = accuracy_score(y_true_total, y_pred_total)
        kappa = cohen_kappa_score(y_true_total, y_pred_total)
        f1 = f1_score(y_true_total, y_pred_total, average='weighted')
        mcc = matthews_corrcoef(y_true_total, y_pred_total)
        
        # Percentage acceptable predict
        substraction = np.abs(y_true_total - y_pred_total)
        total_hits = np.count_nonzero((substraction == 1) | (substraction == 0))
        length_array_prediction = substraction.size
        acceptable_accurary_pct = total_hits / length_array_prediction if length_array_prediction > 0 else 0

        
        # ROC AUC (multiclass, with probabilities)
        try:
            roc_auc = roc_auc_score(y_true_total, y_proba_total, multi_class='ovr',average='weighted')
        except Exception as e:
            roc_auc = None
        
        # Save summary results
        summary_results.append({
            "Model": model_name,
            "Accuracy": acc,
            "Acceptable_accuracy": acceptable_accurary_pct,
            "Kappa_score": kappa,
            "F1_score": f1,
            "MCC": mcc,
            "ROC_AUC": roc_auc
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_results)

    # Save summary CSCV
    summary_filename = f"{config.results_files_dir}{dataset_identifier}_{features_type}_execution_summary.csv"

    # Save CSV
    summary_df.to_csv(summary_filename, index=False,sep=';')
    
    final_result.to_csv(config.results_files_dir+dataset_identifier+"_"+features_type+"_result.csv",sep=';')