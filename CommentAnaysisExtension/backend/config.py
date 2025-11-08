"""
Configuration management for the application.
Loads configuration from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try loading from parent directory as well
    parent_env_path = Path(__file__).parent.parent / '.env'
    if parent_env_path.exists():
        load_dotenv(parent_env_path)


class Config:
    """Application configuration loaded from environment variables."""
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "Comment Analysis API")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
    DEBUG: bool = os.getenv("FLASK_ENV", "production").lower() == "development"
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", os.urandom(32).hex())
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:3000,http://localhost:8080"
    ).split(",")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    MAX_REQUEST_SIZE: int = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    MAX_COMMENTS_PER_REQUEST: int = int(os.getenv("MAX_COMMENTS_PER_REQUEST", "1000"))
    
    # Dagshub Configuration
    DAGSHUB_USERNAME: Optional[str] = os.getenv("DAGSHUB_USERNAME")
    DAGSHUB_TOKEN: Optional[str] = os.getenv("DAGSHUB_TOKEN")
    DAGSHUB_REPO_OWNER: str = os.getenv("DAGSHUB_REPO_OWNER", "AIwithAj")
    DAGSHUB_REPO_NAME: str = os.getenv("DAGSHUB_REPO_NAME", "CommentAnalysis")
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/AIwithAj/CommentAnalysis.mlflow"
    )
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "yt_chrome_plugin_model")
    MLFLOW_VECTORIZER_RUN_ID: str = os.getenv(
        "MLFLOW_VECTORIZER_RUN_ID",
        "e7456a00a6d74d1f9dfc2da425a41d24"
    )
    MLFLOW_ARTIFACT_PATH: str = os.getenv("MLFLOW_ARTIFACT_PATH", "transformer.pkl")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    CLOUDWATCH_LOG_GROUP: str = os.getenv(
        "CLOUDWATCH_LOG_GROUP",
        "/ecs/comment-analysis-backend"
    )
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Health Check
    HEALTH_CHECK_TIMEOUT: int = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not cls.DAGSHUB_USERNAME:
            errors.append("DAGSHUB_USERNAME is required")
        if not cls.DAGSHUB_TOKEN:
            errors.append("DAGSHUB_TOKEN is required")
        if cls.RATE_LIMIT_PER_MINUTE < 1:
            errors.append("RATE_LIMIT_PER_MINUTE must be at least 1")
        if cls.MAX_COMMENTS_PER_REQUEST < 1:
            errors.append("MAX_COMMENTS_PER_REQUEST must be at least 1")
            
        return errors
    
    @classmethod
    @lru_cache()
    def get_config(cls):
        """Get cached configuration instance."""
        return cls()


# Global config instance
config = Config.get_config()

