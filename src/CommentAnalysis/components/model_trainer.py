import sys
import numpy as np
import pandas as pd
from src.CommentAnalysis import logger
import lightgbm as lgb
from scipy import sparse
from src.CommentAnalysis.utils.common import read_data, save_bin, read_yaml
from src.CommentAnalysis.constants import SCHEMA_FILE_PATH


class ModelTrainer:
    def __init__(self, ModelTrainerConfig):
        try:
            self.model_trainer_config = ModelTrainerConfig
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
            logger.info("ModelTrainer initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize ModelTrainer.")
            raise Exception(e, sys)

    def train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float,
        max_depth: int,
        n_estimators: int
    ) -> lgb.LGBMClassifier:
        """
        Train a LightGBM model with specified parameters.
        """
        try:
            logger.info("Starting LightGBM model training...")
            best_model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                metric="multi_logloss",
                is_unbalance=True,
                class_weight="balanced",
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators
            )
            best_model.fit(X_train, y_train)
            logger.info("LightGBM model training completed successfully.")
            return best_model
        except Exception as e:
            logger.exception("Error during LightGBM model training.")
            raise

    def initiate_model_training(self):
        """
        Orchestrate the model training pipeline.
        """
        try:
            logger.info("Loading training data...")
            X_train_tfidf = sparse.load_npz(self.model_trainer_config.x_train_file_path)
            y_train = read_data(self.model_trainer_config.y_train_file_path)

            # Ensure y_train is 1D np.ndarray
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0].values
            elif isinstance(y_train, pd.Series):
                y_train = y_train.values

            logger.info(f"Training data loaded. Shape: X={X_train_tfidf.shape}, y={y_train.shape}")

            model = self.train_lgbm(
                X_train=X_train_tfidf,
                y_train=y_train,
                learning_rate=self.model_trainer_config.learning_rate,
                max_depth=self.model_trainer_config.max_depth,
                n_estimators=self.model_trainer_config.n_estimators
            )

            save_bin(model, self.model_trainer_config.trained_model_path)
            logger.info(f"Trained model saved at: {self.model_trainer_config.trained_model_path}")
        except Exception as e:
            logger.exception("Failed during model training pipeline.")
            raise
