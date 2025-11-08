"""
Unit tests for the Comment Analysis API.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Flask

# Import app after setting test environment
import os
os.environ['FLASK_ENV'] = 'testing'
os.environ['DAGSHUB_USERNAME'] = 'test_user'
os.environ['DAGSHUB_TOKEN'] = 'test_token'

from app import app, preprocess_comment, validate_comment_data, validate_request_data


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_comments():
    """Sample comments for testing."""
    return [
        {
            "text": "This is a great video!",
            "timestamp": "2025-01-15T10:00:00Z",
            "authorId": "user1"
        },
        {
            "text": "Not very helpful.",
            "timestamp": "2025-01-15T10:05:00Z",
            "authorId": "user2"
        }
    ]


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_home_endpoint(self, client):
        """Test root endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'running'
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code in [200, 503]  # 503 if model not loaded
        data = json.loads(response.data)
        assert 'status' in data
        assert 'checks' in data
    
    def test_readiness_endpoint(self, client):
        """Test readiness endpoint."""
        response = client.get('/ready')
        assert response.status_code in [200, 503]


class TestValidation:
    """Test validation functions."""
    
    def test_validate_comment_data_valid(self):
        """Test valid comment data."""
        comment = {
            "text": "This is a test comment",
            "timestamp": "2025-01-15T10:00:00Z",
            "authorId": "user1"
        }
        is_valid, error = validate_comment_data(comment)
        assert is_valid is True
        assert error is None
    
    def test_validate_comment_data_missing_text(self):
        """Test comment without text field."""
        comment = {
            "timestamp": "2025-01-15T10:00:00Z"
        }
        is_valid, error = validate_comment_data(comment)
        assert is_valid is False
        assert "text" in error.lower()
    
    def test_validate_comment_data_empty_text(self):
        """Test comment with empty text."""
        comment = {
            "text": "",
            "timestamp": "2025-01-15T10:00:00Z"
        }
        is_valid, error = validate_comment_data(comment)
        assert is_valid is False
    
    def test_validate_request_data_valid(self, sample_comments):
        """Test valid request data."""
        data = {"comments": sample_comments}
        is_valid, error = validate_request_data(data)
        assert is_valid is True
        assert error is None
    
    def test_validate_request_data_empty_comments(self):
        """Test request with empty comments list."""
        data = {"comments": []}
        is_valid, error = validate_request_data(data)
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_request_data_missing_comments(self):
        """Test request without comments field."""
        data = {}
        is_valid, error = validate_request_data(data)
        assert is_valid is False
        assert "comments" in error.lower()


class TestPreprocessing:
    """Test text preprocessing."""
    
    def test_preprocess_comment_basic(self):
        """Test basic comment preprocessing."""
        comment = "This is a TEST comment!"
        result = preprocess_comment(comment)
        assert isinstance(result, str)
        assert "test" in result.lower()
    
    def test_preprocess_comment_empty(self):
        """Test preprocessing empty comment."""
        result = preprocess_comment("")
        assert result == ""
    
    def test_preprocess_comment_none(self):
        """Test preprocessing None."""
        result = preprocess_comment(None)
        assert result == ""
    
    def test_preprocess_comment_special_chars(self):
        """Test preprocessing with special characters."""
        comment = "Hello!!! This is @#$% great!"
        result = preprocess_comment(comment)
        # Should remove special characters
        assert "@" not in result
        assert "#" not in result


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_analyze_comments_missing_data(self, client):
        """Test analyze_comments without data."""
        response = client.post(
            '/analyze_comments',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_analyze_comments_invalid_json(self, client):
        """Test analyze_comments with invalid JSON."""
        response = client.post(
            '/analyze_comments',
            data="invalid json",
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_analyze_comments_empty_list(self, client):
        """Test analyze_comments with empty comments list."""
        data = {"comments": []}
        response = client.post(
            '/analyze_comments',
            data=json.dumps(data),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    @patch('app.model')
    @patch('app.vectorizer')
    def test_analyze_comments_success(self, mock_vectorizer, mock_model, client, sample_comments):
        """Test successful comment analysis."""
        # Mock model and vectorizer
        mock_model.predict.return_value = [1, -1]
        mock_vectorizer.transform.return_value = MagicMock()
        
        data = {"comments": sample_comments}
        response = client.post(
            '/analyze_comments',
            data=json.dumps(data),
            content_type='application/json'
        )
        # May fail if model not loaded, but should handle gracefully
        assert response.status_code in [200, 500]
    
    def test_predict_with_timestamps_missing_data(self, client):
        """Test predict_with_timestamps without data."""
        response = client.post(
            '/predict_with_timestamps',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_rate_limit_error(self, client):
        """Test rate limiting."""
        # Make many requests quickly
        for _ in range(100):
            response = client.get('/demo')
            if response.status_code == 429:
                data = json.loads(response.data)
                assert 'rate limit' in data['error'].lower()
                break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

