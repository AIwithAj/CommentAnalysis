"""
Model Promotion Script

This script promotes the latest Staging model to Production stage in MLflow.
It should be run after successful model testing to promote validated models.

The script:
1. Connects to MLflow/Dagshub
2. Gets the latest Staging model version
3. Archives current Production model(s)
4. Promotes the Staging model to Production

Usage:
    python scripts/promote_model.py
    
Environment Variables:
    DAGSHUB_TOKEN: Required - Dagshub authentication token
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import dagshub

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
except ImportError:
    pass  # python-dotenv not installed, skip


def promote_model():
    """
    Promote the latest Staging model to Production stage.
    
    This function:
    1. Sets up MLflow connection to Dagshub
    2. Gets the latest Staging model version
    3. Archives existing Production models
    4. Promotes Staging model to Production
    """
    # Set up DagsHub credentials for MLflow tracking
    print("Setting up MLflow connection...")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "AIwithAj"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Initialize Dagshub connection (same as backend and register_model)
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

    # Model configuration (same as project)
    model_name = "yt_chrome_plugin_model"
    
    print(f"\nPromoting model: {model_name}")
    print("=" * 60)
    
    try:
        client = MlflowClient()
        
        # Get the latest version in Staging
        print(f"Fetching latest Staging model version...")
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        
        if not staging_versions:
            raise ValueError(
                f"No model in 'Staging' stage for: {model_name}. "
                f"Make sure a model has been registered to Staging first."
            )
        
        latest_staging = staging_versions[0]
        print(f"✓ Found Staging model version {latest_staging.version}")
        print(f"  Registered at: {latest_staging.last_updated_timestamp}")
        
        # Get current Production models (if any)
        print(f"\nChecking for existing Production models...")
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if prod_versions:
            print(f"Found {len(prod_versions)} Production model(s), archiving...")
            for version in prod_versions:
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=version.version,
                        stage="Archived"
                    )
                    print(f"  ✓ Archived Production model version {version.version}")
                except Exception as e:
                    print(f"  ⚠ Warning: Failed to archive version {version.version}: {e}")
        else:
            print("  No existing Production models found")
        
        # Promote the Staging model to Production
        print(f"\nPromoting Staging model version {latest_staging.version} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging.version,
            stage="Production"
        )
        
        # Also set the Production alias
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=latest_staging.version
            )
            print(f"✓ Set Production alias for version {latest_staging.version}")
        except Exception as e:
            print(f"⚠ Warning: Failed to set Production alias: {e}")
        
        print("=" * 60)
        print(f"✅ SUCCESS: Model version {latest_staging.version} promoted to Production")
        print(f"   Model: {model_name}")
        print(f"   Version: {latest_staging.version}")
        print("=" * 60)
        
    except Exception as e:
        error_msg = (
            f"Failed to promote model: {e}\n"
            f"Model name: {model_name}\n"
            f"Error type: {type(e).__name__}"
        )
        print(f"\n❌ ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    try:
        promote_model()
    except Exception as e:
        print(f"\n❌ Model promotion failed: {e}")
        exit(1)
