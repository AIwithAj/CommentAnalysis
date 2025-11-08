# Comment Analysis Backend API

Production-ready Flask API for YouTube comment sentiment analysis with MLflow integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Dagshub account and token

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials
```

### Running Locally

```bash
# Set credentials
export DAGSHUB_USERNAME="your_username"
export DAGSHUB_TOKEN="your_token"

# Run server
python run_local.py
```

Or use the Makefile:
```bash
make dev-backend
```

## 📋 Configuration

### Required Environment Variables

- `DAGSHUB_USERNAME`: Your Dagshub username
- `DAGSHUB_TOKEN`: Your Dagshub access token

### Optional Environment Variables

- `PORT`: Server port (default: 8000)
- `FLASK_ENV`: Environment (development/production)
- `RATE_LIMIT_PER_MINUTE`: Rate limit (default: 60)
- `ALLOWED_ORIGINS`: CORS allowed origins (comma-separated)

See `.env.example` for all available options.

## 🔌 API Endpoints

### Health Checks
- `GET /` - Root endpoint
- `GET /health` - Health check for load balancer
- `GET /ready` - Readiness check

### Analysis
- `GET /demo` - Demo with sample data
- `POST /analyze_comments` - Full sentiment analysis
- `POST /predict_with_timestamps` - Simple predictions

## 🐳 Docker

### Build
```bash
docker build -t comment-analysis-backend .
```

### Run
```bash
docker run -p 8000:8000 --env-file .env comment-analysis-backend
```

Or use docker-compose from project root:
```bash
docker-compose up
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 📦 Deployment

### AWS ECS Deployment

**Prerequisites:**
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally
- GitHub repository with Actions enabled

**Quick Setup:**

1. **Create ECR Repository:**
   ```bash
   aws ecr create-repository --repository-name comment-analysis-backend --region us-east-1
   ```

2. **Store Secrets in AWS Secrets Manager:**
   ```bash
   aws secretsmanager create-secret --name comment-analysis/dagshub-username --secret-string "your-username"
   aws secretsmanager create-secret --name comment-analysis/dagshub-token --secret-string "your-token"
   ```

3. **Configure GitHub Secrets:**
   - Go to repository Settings → Secrets → Actions
   - Add: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

4. **Update Configuration:**
   - Edit `aws/ecs-task-definition.json` with your AWS account ID
   - Update `aws/cloudformation-template.yaml` with VPC/subnet IDs

5. **Deploy:**
   - Push to main branch triggers CI/CD
   - Monitor deployment in GitHub Actions

**Deployment Architecture:**
```
GitHub → GitHub Actions → ECR → ECS (Fargate) → ALB → Backend Service
```

For detailed CloudFormation deployment, see `aws/cloudformation-template.yaml`.

## 🔒 Security

- ✅ Rate limiting
- ✅ Input validation
- ✅ CORS with specific origins
- ✅ Secrets in environment variables
- ✅ Non-root Docker user
- ✅ Security scanning in CI/CD

## 📊 Monitoring

- Structured logging to stdout
- CloudWatch integration ready
- Health check endpoints
- Request/response logging

## 🛠️ Development

### Code Quality
```bash
# Lint
make lint

# Format
make format

# Type check
mypy .
```

### Project Structure
```
backend/
├── app.py              # Main application
├── config.py           # Configuration management
├── requirements.txt    # Dependencies
├── Dockerfile          # Multi-stage Docker build
├── tests/              # Unit tests
└── aws/                # AWS deployment configs
```

## 📚 Documentation

- [AWS Deployment Guide](./AWS_DEPLOYMENT_GUIDE.md) - Complete AWS setup
- [Project Root README](../../README.md) - Overall project documentation

## 🐛 Troubleshooting

### Model not loading
- Check Dagshub credentials are set
- Verify MLflow tracking URI is correct
- Check network connectivity

### Port already in use
```bash
export PORT=8001
python run_local.py
```

### Import errors
```bash
pip install -r requirements.txt
```

## 📄 License

See LICENSE file in project root.
