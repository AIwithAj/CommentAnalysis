# register model
import json
import mlflow
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
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            alias="Staging"
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
        DAGSHUB_TOKEN=os.getenv('DAGSHUB_TOKEN')
        if not DAGSHUB_TOKEN:
            raise EnvironmentError("DAGSHUB_TOKEN is not set")
        os.environ["MLFLOW_TRACKING_USERNAME"]="AIwithAj"
        os.environ["MLFLOW_TRACKING_PASSWORD"]=DAGSHUB_TOKEN
        mlflow.set_tracking_uri("https://dagshub.com/AIwithAj/CommentAnalysis.mlflow")
        
        config=read_yaml(CONFIG_FILE_PATH).Model_Evaluation
        model_info_path = config.file_path
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()