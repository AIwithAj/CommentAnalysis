import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn
import dagshub

from src.CommentAnalysis.utils.common import load_vectorizer, load_bin, read_data
from src.CommentAnalysis import logger


class Evaluation:
    def __init__(self, config):
        """
        Initialize the Evaluation class with config.
        """
        self.config = config

    def save_model_info(self, run_id: str, model_path: str, file_path: str, metrics: dict = None) -> None:
        """
        Save the model run ID, path and optionally metrics to a JSON file.
        """
        model_info = {'run_id': run_id, 'model_path': model_path}
        if metrics:
            model_info['metrics'] = metrics

        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)

        logger.info(f'Model info and metrics saved to {file_path}')

    def evaluate_model(self, model, X, y):
        """
        Evaluate the model and return classification report & confusion matrix.
        """
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        logger.info('Model evaluation completed')
        return report, cm

    def log_confusion_matrix(self, cm, dataset_name: str):
        """
        Log confusion matrix as an MLflow artifact.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_file_path = self.config.root_dir+f'/confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        logger.info(f'Confusion matrix logged for {dataset_name}')

    def log_metrics(self, report, prefix) -> dict:
        """
        Log metrics from classification report with a prefix.
        Returns a dict of the main metrics for JSON file.
        """
        metrics_summary = {}
        for label, metrics in report.items():
            if label == 'accuracy':
                # accuracy is a float
                mlflow.log_metric(f"{prefix}_accuracy", metrics)
                metrics_summary[f"{prefix}_accuracy"] = metrics
            elif isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"{prefix}_{label}_precision": metrics['precision'],
                    f"{prefix}_{label}_recall": metrics['recall'],
                    f"{prefix}_{label}_f1-score": metrics['f1-score']
                })
                if label == 'weighted avg':
                    metrics_summary[f"{prefix}_precision"] = metrics['precision']
                    metrics_summary[f"{prefix}_recall"] = metrics['recall']
                    metrics_summary[f"{prefix}_f1-score"] = metrics['f1-score']
        logger.info(f'Metrics logged for {prefix} -->{metrics_summary}')
        return metrics_summary


    def log_params_from_yaml(self, yaml_path):
        """
        Load params from params.yaml and log to MLflow.
        """
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f)
        for key, value in params.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    mlflow.log_param(f"{key}_{subkey}", subval)
            else:
                mlflow.log_param(key, value)
        logger.info(f'Parameters from {yaml_path} logged to MLflow')

    def initiate_model_evaluation(self):
        """
        Main function to run model evaluation and log to MLflow.
        """
        dagshub.init(
            repo_owner="AIwithAj",
            repo_name="CommentAnalysis",
            mlflow=True,
        )
        mlflow.set_experiment('dvc-pipeline-runs')

        try:
            with mlflow.start_run() as run:
                logger.info('MLflow run started.')

                # Load artifacts
                model = load_bin(self.config.path_of_model)
                vectorizer = load_vectorizer(self.config.transformer)
                logger.info('Model and vectorizer loaded.')

                X_train_tfidf = sparse.load_npz(self.config.x_train_file_path)
                y_train = read_data(self.config.y_train_file_path)

                test_data = read_data(self.config.testing_data).dropna(subset=['clean_comment', 'category'])
                X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
                y_test = test_data['category'].values

                # Signature & example
                input_example = pd.DataFrame(
                    X_test_tfidf.toarray()[:5],
                    columns=vectorizer.get_feature_names_out()
                )
                signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))
                logger.info('Input example & signature created.')

                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "lgbm_model",
                    signature=signature,
                    input_example=input_example
                )
                logger.info('Model logged to MLflow.')

                # Log vectorizer & params
                mlflow.log_artifact(self.config.transformer)
                self.log_params_from_yaml(self.config.params_file_path)

                # Evaluate & log on Training Data
                train_report, train_cm = self.evaluate_model(model, X_train_tfidf, y_train)
                train_metrics = self.log_metrics(train_report, "train")
                self.log_confusion_matrix(train_cm, "Train Data")

                # Evaluate & log on Test Data
                test_report, test_cm = self.evaluate_model(model, X_test_tfidf, y_test)
                test_metrics = self.log_metrics(test_report, "test")
                self.log_confusion_matrix(test_cm, "Test Data")

                # Save model info with metrics
                model_path = "lgbm_model"
                metrics_summary = {
                    "train": train_metrics,
                    "test": test_metrics
                }
                self.save_model_info(run.info.run_id, model_path, self.config.file_path, metrics=metrics_summary)
                logger.info(f"loggig and saving evaluation metrics successfully--{ self.config.file_path}")
                # MLflow tags
                mlflow.set_tag("model_type", "LightGBM")
                mlflow.set_tag("task", "Sentiment Analysis")
                mlflow.set_tag("dataset", "YouTube Comments")

                logger.info("Model evaluation and logging completed successfully.")

        except Exception as e:
            logger.exception(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")
