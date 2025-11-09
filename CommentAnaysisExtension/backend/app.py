"""
Standalone Comment Analysis API - Flask application
Simplified for demo and deployment
"""
import re
import os
import logging
import time
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import nltk
import mlflow
from mlflow.tracking import MlflowClient
import pickle

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("FLASK_ENV", "production").lower() == "development"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:8080"
).split(",")

# MLflow Configuration
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "AIwithAj")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/AIwithAj/CommentAnalysis.mlflow"
)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "yt_chrome_plugin_model")
MLFLOW_VECTORIZER_RUN_ID = os.getenv(
    "MLFLOW_VECTORIZER_RUN_ID",
    "e7456a00a6d74d1f9dfc2da425a41d24"
)
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "transformer.pkl")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK not available: {e}")
    NLTK_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, origins=ALLOWED_ORIGINS, methods=['GET', 'POST'])

# Global model and vectorizer
model = None
vectorizer = None
model_loaded = False


def load_model_and_vectorizer():
    """Load model and vectorizer from MLflow."""
    global model, vectorizer, model_loaded
    
    if model_loaded:
        return model, vectorizer
    
    try:
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        if not dagshub_token:
            raise EnvironmentError(
                "DAGSHUB_TOKEN not found. Set it as environment variable."
            )
        
        # Set MLflow authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        logger.info("Loading model from MLflow...")
        
        # Load model
        client = MlflowClient()
        latest_prod = client.get_latest_versions(
            MLFLOW_MODEL_NAME,
            stages=["Production"]
        )
        
        if not latest_prod:
            raise ValueError(f"No production model found: {MLFLOW_MODEL_NAME}")
        
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✓ Model loaded")
        
        # Load vectorizer
        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=MLFLOW_VECTORIZER_RUN_ID,
            artifact_path=MLFLOW_ARTIFACT_PATH
        )
        
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info("✓ Vectorizer loaded")
        
        model_loaded = True
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_comment(comment: str) -> str:
    """Preprocess comment text."""
    if not comment or not isinstance(comment, str):
        return ""
    
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    
    if NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english')) - {'not', 'but', 'no'}
            comment = ' '.join([w for w in comment.split() if w not in stop_words])
            
            lemmatizer = WordNetLemmatizer()
            comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        except Exception:
            pass
    
    return comment


def validate_request(data: Dict) -> tuple[bool, Optional[str]]:
    """Validate request data."""
    if not isinstance(data, dict):
        return False, "Request body must be JSON object"
    
    if 'comments' not in data:
        return False, "Missing 'comments' field"
    
    comments = data.get('comments', [])
    if not isinstance(comments, list):
        return False, "'comments' must be an array"
    
    if len(comments) == 0:
        return False, "Comments array is empty"
    
    if len(comments) > 1000:
        return False, "Too many comments (max 1000)"
    
    for i, comment in enumerate(comments):
        if not isinstance(comment, dict):
            return False, f"Comment {i} must be an object"
        
        if 'text' not in comment:
            return False, f"Comment {i} missing 'text' field"
        
        if not isinstance(comment['text'], str):
            return False, f"Comment {i} 'text' must be string"
        
        if len(comment['text']) == 0:
            return False, f"Comment {i} text is empty"
        
        if 'timestamp' not in comment:
            return False, f"Comment {i} missing 'timestamp' field"
    
    return True, None


def analyze_comments(comments_data: List[Dict]) -> Dict:
    """Analyze comments and return results."""
    start_time = time.time()
    
    comments = [item['text'] for item in comments_data]
    timestamps = [item['timestamp'] for item in comments_data]
    author_ids = [item.get('authorId', 'Unknown') for item in comments_data]
    
    # Preprocess and predict
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    predictions = model.predict(transformed)
    
    # Build results
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    sentiment_data = []
    
    for i, (comment, sentiment, timestamp, author_id) in enumerate(
        zip(comments, predictions, timestamps, author_ids)
    ):
        label = "positive" if sentiment == 1 else "neutral" if sentiment == 0 else "negative"
        sentiment_counts[label] += 1
        
        sentiment_data.append({
            "id": i,
            "comment": comment,
            "sentiment": int(sentiment),
            "sentiment_label": label,
            "timestamp": timestamp,
            "author_id": author_id,
            "word_count": len(comment.split())
        })
    
    # Metrics
    total = len(comments)
    unique_commenters = len(set(author_ids))
    total_words = sum(len(c.split()) for c in comments)
    avg_words = round(total_words / total, 2) if total > 0 else 0
    
    avg_sentiment_raw = sum(predictions) / total if total > 0 else 0
    sentiment_score = round(((avg_sentiment_raw + 1) / 2) * 10, 2)
    engagement_score = min(avg_words * 10, 100)
    
    # Hourly trends
    hourly_sentiment = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    
    for item in sentiment_data:
        try:
            dt = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            hour_key = dt.strftime('%Y-%m-%d %H:00:00')
            hourly_sentiment[hour_key][item['sentiment_label']] += 1
        except Exception:
            continue
    
    trend_data = [
        {
            "hour": hour,
            "positive": s["positive"],
            "neutral": s["neutral"],
            "negative": s["negative"]
        }
        for hour, s in sorted(hourly_sentiment.items())
    ]
    
    # Word cloud
    all_words = []
    for comment in preprocessed:
        all_words.extend(comment.split())
    
    word_counts = Counter(all_words)
    word_cloud_data = [[w, c] for w, c in word_counts.most_common(20)]
    
    # Top comments
    positive = [item for item in sentiment_data if item['sentiment_label'] == 'positive']
    negative = [item for item in sentiment_data if item['sentiment_label'] == 'negative']
    most_engaged = sorted(sentiment_data, key=lambda x: x['word_count'], reverse=True)[:10]
    
    processing_time = round(time.time() - start_time, 3)
    
    return {
        "success": True,
        "metrics": {
            "total_comments": total,
            "unique_commenters": unique_commenters,
            "avg_words_per_comment": avg_words,
            "sentiment_score": sentiment_score,
            "engagement_score": engagement_score,
            "processing_time_seconds": processing_time
        },
        "sentiment_distribution": sentiment_counts,
        "sentiment_data": sentiment_data,
        "trend_data": trend_data,
        "word_cloud_data": word_cloud_data,
        "top_comments": {
            "positive": positive[:5],
            "negative": negative[:5],
            "most_engaged": most_engaged
        }
    }


# Routes
@app.route('/')
def home():
    """Root endpoint."""
    return jsonify({
        "message": "Comment Analysis API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/health", "/ready", "/demo", "/analyze_comments", "/predict_with_timestamps"]
    })


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.utcnow().isoformat()
    }), 200 if model_loaded else 503


@app.route('/ready')
def readiness_check():
    """Readiness check endpoint."""
    if model_loaded and model is not None and vectorizer is not None:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503


@app.route('/demo', methods=['GET'])
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
        result = analyze_comments(demo_comments)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return jsonify({"error": "Demo failed", "message": str(e)}), 500


@app.route('/analyze_comments', methods=['POST'])
def analyze_comments_endpoint():
    """Main endpoint for analyzing comments."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        is_valid, error = validate_request(data)
        if not is_valid:
            return jsonify({"error": error}), 400
        
        result = analyze_comments(data['comments'])
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": "Analysis failed"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Simple prediction endpoint."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        is_valid, error = validate_request(data)
        if not is_valid:
            return jsonify({"error": error}), 400
        
        comments_data = data['comments']
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed)
        
        response = [
            {
                "comment": comment,
                "sentiment": int(sentiment),
                "timestamp": timestamp
            }
            for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
        ]
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# Initialize model on startup
try:
    logger.info("Initializing application...")
    load_model_and_vectorizer()
    logger.info("✓ Application ready")
except Exception as e:
    logger.error(f"✗ Initialization failed: {e}")
    logger.warning("App will start but endpoints will not work")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Starting Comment Analysis API")
    logger.info(f"   Environment: {'Development' if DEBUG else 'Production'}")
    logger.info(f"   Host: {HOST}")
    logger.info(f"   Port: {PORT}")
    logger.info("=" * 60)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)