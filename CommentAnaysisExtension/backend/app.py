"""
Comment Analysis API - Production-ready Flask application.
Implements industry best practices for security, logging, and monitoring.
"""
import re
import os
import logging
import time
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import nltk
import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import warnings
from pandas.errors import PerformanceWarning

from config import config

warnings.simplefilter("ignore", PerformanceWarning)

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK download error: {e}")
    NLTK_AVAILABLE = False
    stopwords = None
    WordNetLemmatizer = None

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_REQUEST_SIZE

# Configure CORS with specific origins
CORS(
    app,
    origins=config.ALLOWED_ORIGINS,
    methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization'],
    max_age=3600
)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{config.RATE_LIMIT_PER_MINUTE} per minute"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# Global model and vectorizer (loaded on startup)
model: Optional[Any] = None
vectorizer: Optional[Any] = None
model_loaded: bool = False


def load_model_and_vectorizer() -> tuple:
    """
    Load model and vectorizer from MLflow.
    Uses environment variables for configuration.
    """
    global model, vectorizer, model_loaded
    
    if model_loaded:
        return model, vectorizer
    
    try:
        logger.info("Initializing Dagshub connection...")
        dagshub.init(
            repo_owner=config.DAGSHUB_REPO_OWNER,
            repo_name=config.DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        
        logger.info(f"Loading model: {config.MLFLOW_MODEL_NAME}")
        client = MlflowClient()
        latest_prod = client.get_latest_versions(
            config.MLFLOW_MODEL_NAME,
            stages=["Production"]
        )
        
        if not latest_prod:
            raise ValueError(
                f"No model in 'Production' stage for: {config.MLFLOW_MODEL_NAME}"
            )
        
        model_uri = f"models:/{config.MLFLOW_MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")
        
        logger.info("Downloading vectorizer...")
        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=config.MLFLOW_VECTORIZER_RUN_ID,
            artifact_path=config.MLFLOW_ARTIFACT_PATH
        )
        
        import pickle
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info("Vectorizer loaded successfully")
        
        model_loaded = True
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error loading model/vectorizer: {e}", exc_info=True)
        raise


def preprocess_comment(comment: str) -> str:
    """
    Preprocess a single comment for sentiment analysis.
    
    Args:
        comment: Raw comment text
        
    Returns:
        Preprocessed comment text
    """
    try:
        if not comment or not isinstance(comment, str):
            return ""
        
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        if not NLTK_AVAILABLE:
            # Fallback to basic stopword removal
            common_stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                'at', 'to', 'for', 'of', 'with', 'by'
            }
            comment = ' '.join([
                word for word in comment.split() 
                if word not in common_stopwords
            ])
            return comment
        
        try:
            stop_words = set(stopwords.words('english')) - {
                'not', 'but', 'however', 'no', 'yet'
            }
            comment = ' '.join([
                word for word in comment.split() 
                if word not in stop_words
            ])
        except Exception:
            pass
        
        try:
            lemmatizer = WordNetLemmatizer()
            comment = ' '.join([
                lemmatizer.lemmatize(word) 
                for word in comment.split()
            ])
        except Exception:
            pass
        
        return comment
    except Exception as e:
        logger.warning(f"Error preprocessing comment: {e}")
        return comment if isinstance(comment, str) else ""


def validate_comment_data(comment: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate a single comment object.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(comment, dict):
        return False, "Comment must be a dictionary"
    
    if 'text' not in comment:
        return False, "Comment must have 'text' field"
    
    text = comment.get('text', '')
    if not isinstance(text, str):
        return False, "Comment text must be a string"
    
    if len(text) == 0:
        return False, "Comment text cannot be empty"
    
    if len(text) > 10000:  # Max comment length
        return False, "Comment text too long (max 10000 characters)"
    
    if 'timestamp' not in comment:
        return False, "Comment must have 'timestamp' field"
    
    return True, None


def validate_request_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate the entire request data.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Request body must be a dictionary"
    
    if 'comments' not in data:
        return False, "Request must contain 'comments' field"
    
    comments = data.get('comments', [])
    if not isinstance(comments, list):
        return False, "Comments must be a list"
    
    if len(comments) == 0:
        return False, "Comments list cannot be empty"
    
    if len(comments) > config.MAX_COMMENTS_PER_REQUEST:
        return False, (
            f"Too many comments. Maximum {config.MAX_COMMENTS_PER_REQUEST} "
            "comments per request"
        )
    
    # Validate each comment
    for i, comment in enumerate(comments):
        is_valid, error = validate_comment_data(comment)
        if not is_valid:
            return False, f"Comment {i}: {error}"
    
    return True, None


def analyze_comments_data(comments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze comments and return sentiment analysis results.
    
    Args:
        comments_data: List of comment dictionaries
        
    Returns:
        Analysis results dictionary
    """
    start_time = time.time()
    
    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        author_ids = [item.get('authorId', 'Unknown') for item in comments_data]
        
        # Preprocess comments
        preprocessed_comments = [
            preprocess_comment(comment) 
            for comment in comments
        ]
        
        # Transform and predict
        transformed = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed)
        
        # Process results
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        sentiment_data = []
        
        for i, (comment, sentiment, timestamp, author_id) in enumerate(
            zip(comments, predictions, timestamps, author_ids)
        ):
            if sentiment == 1:
                sentiment_label = "positive"
            elif sentiment == 0:
                sentiment_label = "neutral"
            else:
                sentiment_label = "negative"
            
            sentiment_counts[sentiment_label] += 1
            sentiment_data.append({
                "id": i,
                "comment": comment,
                "sentiment": int(sentiment),
                "sentiment_label": sentiment_label,
                "timestamp": timestamp,
                "author_id": author_id,
                "word_count": len(comment.split())
            })
        
        # Calculate metrics
        total_comments = len(comments)
        unique_commenters = len(set(author_ids))
        total_words = sum(len(comment.split()) for comment in comments)
        avg_words_per_comment = (
            round(total_words / total_comments, 2) 
            if total_comments > 0 else 0
        )
        
        total_sentiment_score = sum(predictions)
        avg_sentiment_raw = (
            total_sentiment_score / total_comments 
            if total_comments > 0 else 0
        )
        avg_sentiment_normalized = round(((avg_sentiment_raw + 1) / 2) * 10, 2)
        engagement_score = min(avg_words_per_comment * 10, 100)
        
        # Hourly sentiment trends
        hourly_sentiment = defaultdict(
            lambda: {"positive": 0, "neutral": 0, "negative": 0}
        )
        
        for item in sentiment_data:
            try:
                timestamp_dt = datetime.fromisoformat(
                    item['timestamp'].replace('Z', '+00:00')
                )
                hour_key = timestamp_dt.strftime('%Y-%m-%d %H:00:00')
                sentiment_label = item['sentiment_label']
                hourly_sentiment[hour_key][sentiment_label] += 1
            except Exception:
                continue
        
        trend_data = [{
            "hour": hour,
            "positive": sentiments["positive"],
            "neutral": sentiments["neutral"],
            "negative": sentiments["negative"]
        } for hour, sentiments in sorted(hourly_sentiment.items())]
        
        # Word cloud data
        all_words = []
        for comment in preprocessed_comments:
            all_words.extend(comment.split())
        
        word_counts = Counter(all_words)
        word_cloud_data = [
            [word, count] 
            for word, count in word_counts.most_common(20)
        ]
        
        # Top comments
        positive_comments = [
            item for item in sentiment_data 
            if item['sentiment_label'] == 'positive'
        ]
        negative_comments = [
            item for item in sentiment_data 
            if item['sentiment_label'] == 'negative'
        ]
        most_engaged = sorted(
            sentiment_data, 
            key=lambda x: x['word_count'], 
            reverse=True
        )[:10]
        
        processing_time = round(time.time() - start_time, 3)
        
        return {
            "success": True,
            "metrics": {
                "total_comments": total_comments,
                "unique_commenters": unique_commenters,
                "avg_words_per_comment": avg_words_per_comment,
                "sentiment_score": avg_sentiment_normalized,
                "engagement_score": engagement_score,
                "processing_time_seconds": processing_time
            },
            "sentiment_distribution": sentiment_counts,
            "sentiment_data": sentiment_data,
            "trend_data": trend_data,
            "word_cloud_data": word_cloud_data,
            "top_comments": {
                "positive": positive_comments[:5],
                "negative": negative_comments[:5],
                "most_engaged": most_engaged
            }
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_comments_data: {e}", exc_info=True)
        raise


# Request logging middleware
@app.before_request
def log_request_info():
    """Log incoming requests."""
    logger.info(
        f"{request.method} {request.path} - "
        f"IP: {get_remote_address()}"
    )


@app.after_request
def log_response_info(response):
    """Log response information."""
    logger.info(
        f"{request.method} {request.path} - "
        f"Status: {response.status_code}"
    )
    return response


# Routes
@app.route('/')
def home():
    """Root endpoint."""
    return jsonify({
        "message": config.APP_NAME,
        "status": "running",
        "version": "1.0.0"
    })


@app.route('/health')
def health_check():
    """Health check endpoint for load balancer."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "model_loaded": model_loaded,
            "nltk_available": NLTK_AVAILABLE,
            "sentiment_engine": "mlflow_model"
        }
    }
    
    # Check if model is loaded
    if not model_loaded:
        health_status["status"] = "degraded"
        health_status["message"] = "Model not loaded"
        return jsonify(health_status), 503
    
    return jsonify(health_status), 200


@app.route('/ready')
def readiness_check():
    """Readiness check endpoint."""
    if model_loaded and model is not None and vectorizer is not None:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503


@app.route('/demo', methods=['GET'])
@limiter.limit("10 per minute")
def demo_analysis():
    """Demo endpoint with sample data."""
    demo_comments = [
        {
            "text": "This video is absolutely amazing!",
            "timestamp": "2025-01-15T10:00:00Z",
            "authorId": "user1"
        },
        {
            "text": "This is terrible. Waste of my time.",
            "timestamp": "2025-01-15T10:45:00Z",
            "authorId": "user2"
        },
        {
            "text": "Perfect explanation! Thank you",
            "timestamp": "2025-01-15T11:00:00Z",
            "authorId": "user3"
        },
    ]
    try:
        return jsonify(analyze_comments_data(demo_comments))
    except Exception as e:
        logger.error(f"Error in demo_analysis: {e}", exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500


@app.route('/analyze_comments', methods=['POST'])
@limiter.limit("30 per minute")
def analyze_comments():
    """Main endpoint for analyzing comments."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        # Validate request data
        is_valid, error = validate_request_data(data)
        if not is_valid:
            return jsonify({"error": error}), 400
        
        comments_data = data.get('comments')
        
        # Analyze comments
        result = analyze_comments_data(comments_data)
        return jsonify(result), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in analyze_comments: {e}", exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "message": "Internal server error"
        }), 500


@app.route('/predict_with_timestamps', methods=['POST'])
@limiter.limit("30 per minute")
def predict_with_timestamps():
    """Endpoint for simple predictions with timestamps."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        # Validate request data
        is_valid, error = validate_request_data(data)
        if not is_valid:
            return jsonify({"error": error}), 400
        
        comments_data = data.get('comments')
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        
        # Preprocess and predict
        preprocessed_comments = [
            preprocess_comment(comment) 
            for comment in comments
        ]
        transformed = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed)
        predictions = [int(p) for p in predictions]
        
        response = [
            {
                "comment": comment,
                "sentiment": sentiment,
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp in zip(
                comments, predictions, timestamps
            )
        ]
        return jsonify(response), 200
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in predict_with_timestamps: {e}", exc_info=True)
        return jsonify({
            "error": "Prediction failed",
            "message": "Internal server error"
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors."""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": f"Too many requests. Limit: {config.RATE_LIMIT_PER_MINUTE} per minute"
    }), 429


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request too large errors."""
    return jsonify({
        "error": "Request too large",
        "message": f"Maximum request size: {config.MAX_REQUEST_SIZE} bytes"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


# Initialize model on startup
def initialize_model():
    """Load model and vectorizer on startup."""
    try:
        logger.info("Initializing model and vectorizer...")
        load_model_and_vectorizer()
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        # Don't raise - allow app to start but health check will fail


# Initialize model when app starts
initialize_model()


if __name__ == '__main__':
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.warning("Continuing with defaults...")
    
    # Load model
    try:
        load_model_and_vectorizer()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Application will start but may not function correctly")
    
    # Run application
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
