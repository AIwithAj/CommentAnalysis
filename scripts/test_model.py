"""
Model Testing Script for Staging Models

This script tests the model registered to MLflow Staging stage by the CI workflow.
The CI workflow runs the DVC pipeline (stage 6: register_model) which registers
the model to Staging stage. This test suite validates:
- Model loading from Staging stage
- Vectorizer loading
- Model signature validation
- Performance metrics on test data
- Batch prediction capabilities

Usage:
    python scripts/test_model.py
    or
    python -m pytest scripts/test_model.py -v
"""
import unittest
import mlflow
import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from pathlib import Path
import dagshub

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip


class TestModelLoading(unittest.TestCase):
    """
    Test suite for validating the Staging model registered by CI workflow.
    
    This test suite loads the model from MLflow Staging stage (registered by 
    stage 6 pipeline in CI workflow) and validates:
    - Model loading and vectorizer loading
    - Model signature and input/output validation
    - Model performance on test data
    - Batch prediction capabilities
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test fixtures - load Staging model and vectorizer from MLflow.
        
        Loads the model from Staging stage which is registered by CI workflow
        (stage 6: register_model pipeline).
        """
        # Set up DagsHub credentials for MLflow tracking
        print("Setting up MLflow connection...")
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "AIwithAj"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Initialize Dagshub connection (same as backend)
        print("Initializing Dagshub connection...")
        try:
            dagshub.init(
                repo_owner="AIwithAj",
                repo_name="CommentAnalysis",
                mlflow=True,
            )
            print("✓ Dagshub initialized")
        except Exception as e:
            print(f"Warning: Dagshub init failed (may still work): {e}")

        # Set up MLflow tracking URI
        print("Setting MLflow tracking URI...")
        mlflow.set_tracking_uri("https://dagshub.com/AIwithAj/CommentAnalysis.mlflow")
        print("✓ MLflow tracking URI set")

        # Model configuration (same as backend)
        cls.model_name = "yt_chrome_plugin_model"
        cls.vectorizer_run_id = os.getenv(
            "MLFLOW_VECTORIZER_RUN_ID",
            "e7456a00a6d74d1f9dfc2da425a41d24"
        )
        cls.artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH", "transformer.pkl")

        # Load the model from MLflow Staging stage (registered by CI workflow stage 6)
        print(f"Connecting to MLflow and fetching Staging model: {cls.model_name}...")
        try:
            start_time = time.time()
            client = mlflow.tracking.MlflowClient()
            print("  MLflow client created, fetching model versions...")
            latest_staging = client.get_latest_versions(
                cls.model_name,
                stages=["Staging"]
            )
            elapsed = time.time() - start_time
            print(f"  ✓ Fetched model versions in {elapsed:.2f}s")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to MLflow or fetch model versions: {e}\n"
                f"Check your DAGSHUB_TOKEN and network connection.\n"
                f"Error type: {type(e).__name__}"
            ) from e
        
        if not latest_staging:
            raise ValueError(
                f"No model in 'Staging' stage for: {cls.model_name}. "
                f"Make sure the CI workflow has run and registered a model to Staging."
            )

        print(f"Found Staging model version {latest_staging[0].version}")
        model_uri = f"models:/{cls.model_name}/Staging"
        
        print("Loading model from MLflow...")
        try:
            start_time = time.time()
            cls.model = mlflow.sklearn.load_model(model_uri)
            elapsed = time.time() - start_time
            print(f"✓ Loaded model: {cls.model_name} from Staging stage (version {latest_staging[0].version}) in {elapsed:.2f}s")
            print(f"  Model registered at: {latest_staging[0].last_updated_timestamp}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from MLflow: {e}\n"
                f"Model URI: {model_uri}\n"
                f"Error type: {type(e).__name__}"
            ) from e

        # Load the vectorizer from MLflow (same as backend)
        print(f"Downloading vectorizer from run_id: {cls.vectorizer_run_id}...")
        try:
            start_time = time.time()
            vectorizer_path = mlflow.artifacts.download_artifacts(
                run_id=cls.vectorizer_run_id,
                artifact_path=cls.artifact_path
            )
            download_time = time.time() - start_time
            print(f"  Downloaded in {download_time:.2f}s, loading vectorizer...")
            
            with open(vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)
            total_time = time.time() - start_time
            print(f"✓ Vectorizer loaded successfully from {vectorizer_path} (total: {total_time:.2f}s)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download or load vectorizer: {e}\n"
                f"Run ID: {cls.vectorizer_run_id}, Artifact: {cls.artifact_path}\n"
                f"Error type: {type(e).__name__}"
            ) from e

        # Load test data from project artifacts
        test_data_path = Path("artifacts/data_validation/test_data.csv")
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
        
        cls.test_data = pd.read_csv(test_data_path)
        print(f"Loaded test data: {len(cls.test_data)} samples from {test_data_path}")

    def test_model_loaded_properly(self):
        """Test that the model is loaded and not None."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.vectorizer)

    def test_vectorizer_loaded_properly(self):
        """Test that the vectorizer is loaded and has expected methods."""
        self.assertIsNotNone(self.vectorizer)
        self.assertTrue(hasattr(self.vectorizer, 'transform'))
        self.assertTrue(hasattr(self.vectorizer, 'get_feature_names_out'))

    def test_model_signature(self):
        """Test model input/output signature with a sample text."""
        # Create a dummy input for the model
        input_text = "This is a test comment for sentiment analysis"
        
        # Transform using vectorizer (same as backend preprocessing)
        input_transformed = self.vectorizer.transform([input_text])
        
        # Predict using the model (same as backend)
        prediction = self.model.predict(input_transformed)
        
        # Convert to int (same as backend does)
        prediction = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)
        
        # Verify the output shape
        self.assertIn(prediction, [0, 1])  # Binary classification: 0 (neutral/negative) or 1 (positive)

    def test_model_performance(self):
        """Test model performance on holdout test data."""
        # Check if test data has required columns
        required_columns = ['clean_comment', 'category']
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]
        if missing_columns:
            self.skipTest(f"Test data missing required columns: {missing_columns}")
        
        # Prepare test data (same as evaluation stage)
        test_data_clean = self.test_data.dropna(subset=['clean_comment', 'category'])
        
        if len(test_data_clean) == 0:
            self.skipTest("No valid test data after cleaning")
        
        # Extract features and labels
        X_test = test_data_clean['clean_comment'].values
        y_test = test_data_clean['category'].values
        
        # Transform using vectorizer (same as backend)
        X_test_transformed = self.vectorizer.transform(X_test)
        
        # Predict using the model (same as backend)
        y_pred = self.model.predict(X_test_transformed)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nModel Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Test samples: {len(y_test)}")
        
        # Define expected thresholds for the performance metrics
        # Adjust these based on your model's expected performance
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40
        
        # Assert that the model meets the performance thresholds
        self.assertGreaterEqual(
            accuracy, expected_accuracy,
            f'Accuracy should be at least {expected_accuracy}, got {accuracy:.4f}'
        )
        self.assertGreaterEqual(
            precision, expected_precision,
            f'Precision should be at least {expected_precision}, got {precision:.4f}'
        )
        self.assertGreaterEqual(
            recall, expected_recall,
            f'Recall should be at least {expected_recall}, got {recall:.4f}'
        )
        self.assertGreaterEqual(
            f1, expected_f1,
            f'F1 score should be at least {expected_f1}, got {f1:.4f}'
        )

    def test_batch_prediction(self):
        """Test batch prediction similar to backend API."""
        # Sample comments for batch prediction
        sample_comments = [
            "This video is absolutely amazing!",
            "This is terrible. Waste of my time.",
            "Perfect explanation! Thank you",
            "Not bad, could be better",
            "Love this content!"
        ]
        
        # Transform using vectorizer (same as backend)
        transformed = self.vectorizer.transform(sample_comments)
        
        # Predict using the model (same as backend)
        predictions = self.model.predict(transformed)
        
        # Convert to int list (same as backend does - line 551 in app.py)
        predictions = [int(p) for p in predictions]
        
        # Verify predictions
        self.assertEqual(len(predictions), len(sample_comments))
        self.assertTrue(all(pred in [0, 1] for pred in predictions), 
                       f"Predictions should be 0 or 1, got: {predictions}")
        
        print(f"\nBatch Prediction Results:")
        for comment, pred in zip(sample_comments, predictions):
            sentiment = "positive" if pred == 1 else "neutral/negative"
            print(f"  '{comment[:50]}...' -> {sentiment} ({pred})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
