import json
import sys
import pandas as pd
from pandas import DataFrame
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split
from src.CommentAnalysis import logger
from src.CommentAnalysis.constants import SCHEMA_FILE_PATH
from src.CommentAnalysis.utils.common import save_json, read_yaml


class DataValidation:
    def __init__(self, data_validation_config):
        try:
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise Exception(e, sys)

    def validate_number_of_columns(self, df: DataFrame) -> bool:
        try:
            expected_cols = len(self._schema_config["columns"])
            status = len(df.columns) == expected_cols
            logger.info(f"Is required column count correct: {status}")
            return status
        except Exception as e:
            raise Exception(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            missing_cols = [
                col for col in self._schema_config["categorical_columns"]
                if col not in df.columns
            ]
            if missing_cols:
                logger.info(f"Missing categorical columns: {missing_cols}")
                return False
            return True
        except Exception as e:
            raise Exception(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Run drift detection and save report. Return True if drift detected, False otherwise.
        """
        try:
            report = Report(metrics=[DataDriftPreset()])
            data_drift=report.run(reference_data=reference_df, current_data=current_df)

            data_drift.save_json(self.data_validation_config.drift_report_file_path)
            logger.info(f"Drift report saved at {self.data_validation_config.drift_report_file_path}")

            with open(self.data_validation_config.drift_report_file_path) as f:
                json_report = json.load(f)

            # robust check
            drift_detected = False
            for metric in json_report['metrics']:
                if 'DriftedColumnsCount' in metric['metric_id']:
                    count = metric['value'].get('count', 0)
                    share = metric['value'].get('share', 0.0)
                    logger.info(f"Drifted columns: {count} ({share*100:.2f}%)")
                    if count > 0 or share > 0:
                        drift_detected = True
            logger.info(f"Dataset drift detected: {drift_detected}")
            return drift_detected

        except Exception as e:
            raise Exception(e, sys)

    def initiate_data_validation(self) -> dict:
        """
        Run the entire validation workflow.
        """
        try:
            validation_error_msg = ""
            logger.info("Starting data validation")

            df = self.read_data(self.data_validation_config.local_file_path)

            if not self.validate_number_of_columns(df):
                validation_error_msg += "Column count mismatch. "

            if not self.is_column_exist(df):
                validation_error_msg += "Missing required columns. "

            validation_status = len(validation_error_msg) == 0

            train_path, test_path = None, None

            if validation_status:
                reference_df = df.copy()
                current_df = df.copy()

                drift_status = self.detect_dataset_drift(reference_df, current_df)

                if drift_status:
                    validation_error_msg += "Drift detected. "
                    validation_status = False
                else:
                    logger.info("No significant drift detected.")

                if validation_status:
                    train_df, test_df = train_test_split(
                        df,
                        test_size=self.data_validation_config.test_size,
                        random_state=42,
                        stratify=df[self._schema_config["categorical_columns"][0]]
                    )
                    train_path = self.data_validation_config.train_file_path
                    test_path = self.data_validation_config.test_file_path

                    train_df.to_csv(train_path, index=False)
                    test_df.to_csv(test_path, index=False)

                    logger.info(f"Train file saved to: {train_path}")
                    logger.info(f"Test file saved to: {test_path}")
                else:
                    logger.info("Validation failed → skipping train/test split.")
            else:
                logger.info(f"Validation failed: {validation_error_msg}")

            validation_dict = {
                "validation_status": validation_status,
                "validation_message": validation_error_msg,
                "valid_train_file_path": train_path,
                "valid_test_file_path": test_path,
                "drift_report_file_path": self.data_validation_config.drift_report_file_path,
            }

            save_json(self.data_validation_config.validation_status_file, validation_dict)
            return validation_dict

        except Exception as e:
            raise Exception(e, sys)
