# register model
import json
import mlflow
from datetime import datetime
from src.CommentAnalysis import logger
from src.CommentAnalysis.constants import CONFIG_FILE_PATH
from src.CommentAnalysis.utils.common import read_yaml
import dagshub
# Set up MLflow tracking URI


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        # Set alias for the model version
        client.set_registered_model_alias(
            name=model_name,
            alias="Staging",
            version=model_version.version
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        # dagshub.init(
        #     repo_owner="AIwithAj",
        #     repo_name="CommentAnalysis",
        #     mlflow=True,
        # )
        
        import os
        from pathlib import Path
        
        # Try to load from .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent.parent.parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f'Loaded environment variables from {env_path}')
        except ImportError:
            pass  # python-dotenv not installed, skip
        
        DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
        
        config = read_yaml(CONFIG_FILE_PATH).Model_Evaluation
        model_info_path = config.file_path
        model_info = load_model_info(model_info_path)
        model_name = "yt_chrome_plugin_model"
        
        # Only register to MLflow if token is available
        if DAGSHUB_TOKEN:
            os.environ["MLFLOW_TRACKING_USERNAME"] = "AIwithAj"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
            mlflow.set_tracking_uri("https://dagshub.com/AIwithAj/CommentAnalysis.mlflow")
            
            try:
                register_model(model_name, model_info)
                logger.info(f'Model {model_name} registered successfully to MLflow')
            except Exception as e:
                logger.warning(f'Failed to register model to MLflow: {e}. Continuing to create marker file.')
        else:
            logger.warning('DAGSHUB_TOKEN is not set. Skipping MLflow registration. Set DAGSHUB_TOKEN environment variable to register model.')
        
        # Create marker file to track successful registration (or skipped registration)
        marker_file = Path(config.root_dir) / "model_registered.txt"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(marker_file, 'w') as f:
            if DAGSHUB_TOKEN:
                f.write(f"Model {model_name} registered successfully\n")
            else:
                f.write(f"Model {model_name} registration skipped (DAGSHUB_TOKEN not set)\n")
            f.write(f"Run ID: {model_info.get('run_id', 'N/A')}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        logger.info(f'Model registration marker file created: {marker_file}')
        
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()