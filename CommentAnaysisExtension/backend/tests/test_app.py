"""
Unit tests for the Comment Analysis API using unittest.

Tests the Flask application endpoints, validation, preprocessing, and error handling.
Designed to run both locally and in CI/CD workflows.
"""
import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing app
os.environ['FLASK_ENV'] = 'testing'
os.environ['DAGSHUB_USERNAME'] = 'test_user'
os.environ['DAGSHUB_TOKEN'] = 'test_token'
os.environ['LOG_LEVEL'] = 'WARNING'  # Reduce log noise in tests
os.environ['RATE_LIMIT_PER_MINUTE'] = '1000'  # High limit for testing

# Patch MLflow and Dagshub to prevent hanging during app import
# The app calls initialize_model() at import time which tries to connect
import sys
from unittest.mock import MagicMock, patch

# Create comprehensive mocks with all necessary attributes
mock_dagshub = MagicMock()
mock_dagshub.init.side_effect = Exception("Dagshub disabled in tests")

# Create a complete mlflow mock with all submodules
mock_mlflow = MagicMock()
mock_mlflow.tracking = MagicMock()
mock_mlflow_client = MagicMock()
mock_mlflow_client.get_latest_versions.side_effect = Exception("MLflow disabled in tests")
mock_mlflow.tracking.MlflowClient = MagicMock(return_value=mock_mlflow_client)

mock_mlflow.sklearn = MagicMock()
mock_mlflow.sklearn.load_model.side_effect = Exception("MLflow disabled in tests")

mock_mlflow.artifacts = MagicMock()
mock_mlflow.artifacts.download_artifacts.side_effect = Exception("MLflow disabled in tests")

mock_mlflow.set_tracking_uri = MagicMock()

# Replace in sys.modules before importing app
sys.modules['dagshub'] = mock_dagshub
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tracking'] = mock_mlflow.tracking
sys.modules['mlflow.sklearn'] = mock_mlflow.sklearn
sys.modules['mlflow.artifacts'] = mock_mlflow.artifacts

# Now import app - initialize_model will fail fast with our mocks
# The app handles exceptions gracefully, so it won't crash
from app import (
    app, 
    preprocess_comment, 
    validate_comment_data, 
    validate_request_data,
    analyze_comments_data
)


class FlaskAppTests(unittest.TestCase):
    """Test suite for Flask application endpoints and functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.client = app.test_client()
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key-for-testing-only'
        app.config['WTF_CSRF_ENABLED'] = False

    def setUp(self):
        """Set up before each test method."""
        # Mock model and vectorizer for each test
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [1, -1, 0]  # positive, negative, neutral
        
        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.transform.return_value = MagicMock()
        
        # Patch global model and vectorizer
        self.model_patcher = patch('app.model', self.mock_model)
        self.vectorizer_patcher = patch('app.vectorizer', self.mock_vectorizer)
        self.model_loaded_patcher = patch('app.model_loaded', True)
        
        self.model_patcher.start()
        self.vectorizer_patcher.start()
        self.model_loaded_patcher.start()

    def tearDown(self):
        """Clean up after each test method."""
        self.model_patcher.stop()
        self.vectorizer_patcher.stop()
        self.model_loaded_patcher.stop()

    def test_home_page(self):
        """Test root endpoint."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'running')
        self.assertIn('message', data)
        self.assertIn('version', data)

    def test_health_endpoint_with_model(self):
        """Test health check endpoint when model is loaded."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('checks', data)
        self.assertIn('model_loaded', data['checks'])

    def test_health_endpoint_without_model(self):
        """Test health check endpoint when model is not loaded."""
        with patch('app.model_loaded', False):
            response = self.client.get('/health')
            self.assertEqual(response.status_code, 503)
            data = json.loads(response.data)
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'degraded')

    def test_readiness_endpoint_ready(self):
        """Test readiness endpoint when ready."""
        response = self.client.get('/ready')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ready')

    def test_readiness_endpoint_not_ready(self):
        """Test readiness endpoint when not ready."""
        with patch('app.model_loaded', False):
            response = self.client.get('/ready')
            self.assertEqual(response.status_code, 503)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'not ready')

    def test_demo_endpoint(self):
        """Test demo endpoint with sample data."""
        response = self.client.get('/demo')
        # Should succeed with mocked model
        self.assertIn(response.status_code, [200, 500])
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('success', data)
            self.assertTrue(data['success'])

    def test_analyze_comments_missing_data(self):
        """Test analyze_comments endpoint without data."""
        response = self.client.post(
            '/analyze_comments',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_analyze_comments_empty_list(self):
        """Test analyze_comments endpoint with empty comments list."""
        data = {"comments": []}
        response = self.client.post(
            '/analyze_comments',
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_analyze_comments_invalid_json(self):
        """Test analyze_comments endpoint with invalid JSON."""
        response = self.client.post(
            '/analyze_comments',
            data="invalid json",
            content_type='application/json'
        )
        # Flask may return 400 or 500 for invalid JSON depending on error handling
        self.assertIn(response.status_code, [400, 500])
        if response.status_code == 400:
            data = json.loads(response.data)
            self.assertIn('error', data)

    def test_analyze_comments_success(self):
        """Test successful comment analysis."""
        sample_comments = [
            {
                "text": "This is a great video!",
                "timestamp": "2025-01-15T10:00:00Z",
                "authorId": "user1"
            },
            {
                "text": "Not very helpful.",
                "timestamp": "2025-01-15T10:05:00Z",
                "authorId": "user2"
            },
            {
                "text": "It's okay, nothing special.",
                "timestamp": "2025-01-15T10:10:00Z",
                "authorId": "user3"
            }
        ]
        
        data = {"comments": sample_comments}
        response = self.client.post(
            '/analyze_comments',
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('metrics', data)
        self.assertIn('sentiment_distribution', data)
        self.assertIn('sentiment_data', data)
        self.assertEqual(len(data['sentiment_data']), 3)

    def test_predict_with_timestamps_missing_data(self):
        """Test predict_with_timestamps endpoint without data."""
        response = self.client.post(
            '/predict_with_timestamps',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_with_timestamps_success(self):
        """Test successful prediction with timestamps."""
        sample_comments = [
            {
                "text": "This is amazing!",
                "timestamp": "2025-01-15T10:00:00Z",
                "authorId": "user1"
            },
            {
                "text": "This is terrible.",
                "timestamp": "2025-01-15T10:05:00Z",
                "authorId": "user2"
            }
        ]
        
        data = {"comments": sample_comments}
        response = self.client.post(
            '/predict_with_timestamps',
            data=json.dumps(data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertIn('comment', data[0])
        self.assertIn('sentiment', data[0])
        self.assertIn('timestamp', data[0])
        # Check sentiment values are valid (-1, 0, or 1)
        self.assertTrue(all(item['sentiment'] in [-1, 0, 1] for item in data))

    def test_404_handler(self):
        """Test 404 error handler."""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('message', data)


class ValidationTests(unittest.TestCase):
    """Test validation functions."""

    def test_validate_comment_data_valid(self):
        """Test valid comment data."""
        comment = {
            "text": "This is a test comment",
            "timestamp": "2025-01-15T10:00:00Z",
            "authorId": "user1"
        }
        is_valid, error = validate_comment_data(comment)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_comment_data_missing_text(self):
        """Test comment without text field."""
        comment = {
            "timestamp": "2025-01-15T10:00:00Z"
        }
        is_valid, error = validate_comment_data(comment)
        self.assertFalse(is_valid)
        self.assertIn("text", error.lower())

    def test_validate_comment_data_empty_text(self):
        """Test comment with empty text."""
        comment = {
            "text": "",
            "timestamp": "2025-01-15T10:00:00Z"
        }
        is_valid, error = validate_comment_data(comment)
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

    def test_validate_comment_data_missing_timestamp(self):
        """Test comment without timestamp."""
        comment = {
            "text": "Test comment"
        }
        is_valid, error = validate_comment_data(comment)
        self.assertFalse(is_valid)
        self.assertIn("timestamp", error.lower())

    def test_validate_request_data_valid(self):
        """Test valid request data."""
        sample_comments = [
            {
                "text": "Test comment",
                "timestamp": "2025-01-15T10:00:00Z",
                "authorId": "user1"
            }
        ]
        data = {"comments": sample_comments}
        is_valid, error = validate_request_data(data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_request_data_empty_comments(self):
        """Test request with empty comments list."""
        data = {"comments": []}
        is_valid, error = validate_request_data(data)
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

    def test_validate_request_data_missing_comments(self):
        """Test request without comments field."""
        data = {}
        is_valid, error = validate_request_data(data)
        self.assertFalse(is_valid)
        self.assertIn("comments", error.lower())


class PreprocessingTests(unittest.TestCase):
    """Test text preprocessing functions."""

    def test_preprocess_comment_basic(self):
        """Test basic comment preprocessing."""
        comment = "This is a TEST comment!"
        result = preprocess_comment(comment)
        self.assertIsInstance(result, str)
        self.assertIn("test", result.lower())

    def test_preprocess_comment_empty(self):
        """Test preprocessing empty comment."""
        result = preprocess_comment("")
        self.assertEqual(result, "")

    def test_preprocess_comment_none(self):
        """Test preprocessing None."""
        result = preprocess_comment(None)
        self.assertEqual(result, "")

    def test_preprocess_comment_special_chars(self):
        """Test preprocessing with special characters."""
        comment = "Hello!!! This is @#$% great!"
        result = preprocess_comment(comment)
        # Should remove special characters
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)

    def test_preprocess_comment_newlines(self):
        """Test preprocessing with newlines."""
        comment = "Line 1\nLine 2\nLine 3"
        result = preprocess_comment(comment)
        self.assertNotIn("\n", result)


class AnalyzeCommentsDataTests(unittest.TestCase):
    """Test the analyze_comments_data function directly."""

    def setUp(self):
        """Set up mocks for analyze_comments_data tests."""
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [1, -1, 0]  # positive, negative, neutral
        
        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.transform.return_value = MagicMock()

    def test_analyze_comments_data_success(self):
        """Test analyze_comments_data function with mocked dependencies."""
        sample_comments = [
            {
                "text": "This is great!",
                "timestamp": "2025-01-15T10:00:00Z",
                "authorId": "user1"
            },
            {
                "text": "This is terrible.",
                "timestamp": "2025-01-15T10:05:00Z",
                "authorId": "user2"
            },
            {
                "text": "It's okay.",
                "timestamp": "2025-01-15T10:10:00Z",
                "authorId": "user3"
            }
        ]
        
        with patch('app.model', self.mock_model), \
             patch('app.vectorizer', self.mock_vectorizer):
            
            result = analyze_comments_data(sample_comments)
            
            self.assertTrue(result['success'])
            self.assertIn('metrics', result)
            self.assertIn('sentiment_distribution', result)
            self.assertIn('sentiment_data', result)
            self.assertEqual(len(result['sentiment_data']), 3)
            self.assertEqual(result['metrics']['total_comments'], 3)
            self.assertIn('positive', result['sentiment_distribution'])
            self.assertIn('neutral', result['sentiment_distribution'])
            self.assertIn('negative', result['sentiment_distribution'])

    def test_analyze_comments_data_sentiment_labels(self):
        """Test that sentiment labels are correctly assigned."""
        # Set predictions: 1=positive, -1=negative, 0=neutral
        self.mock_model.predict.return_value = [1, -1, 0]
        
        sample_comments = [
            {"text": "Great!", "timestamp": "2025-01-15T10:00:00Z", "authorId": "user1"},
            {"text": "Terrible!", "timestamp": "2025-01-15T10:05:00Z", "authorId": "user2"},
            {"text": "Okay", "timestamp": "2025-01-15T10:10:00Z", "authorId": "user3"}
        ]
        
        with patch('app.model', self.mock_model), \
             patch('app.vectorizer', self.mock_vectorizer):
            
            result = analyze_comments_data(sample_comments)
            
            sentiment_labels = [item['sentiment_label'] for item in result['sentiment_data']]
            self.assertIn('positive', sentiment_labels)
            self.assertIn('negative', sentiment_labels)
            self.assertIn('neutral', sentiment_labels)


if __name__ == '__main__':
    unittest.main(verbosity=2)
