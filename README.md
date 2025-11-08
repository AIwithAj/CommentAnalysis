# Comment Analysis - End-to-End MLOps Project

A complete MLOps pipeline for YouTube comment sentiment analysis with production-ready deployment.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Prerequisites](#-prerequisites)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Backend](#-running-the-backend)
- [Running the Frontend](#-running-the-frontend)
- [Running the Complete Project](#-running-the-complete-project)
- [ML Pipeline](#-ml-pipeline)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)

## 🎯 Project Overview

This project consists of three main components:
1. **ML Pipeline** - End-to-end MLOps pipeline for training sentiment analysis models
2. **Backend API** - Flask REST API for sentiment analysis
3. **Frontend Extension** - Chrome extension built with Vue.js for YouTube comment analysis

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16+** - [Download Node.js](https://nodejs.org/)
- **npm** or **yarn** - Comes with Node.js
- **Git** - [Download Git](https://git-scm.com/downloads)
- **DVC** (Optional, for ML pipeline) - `pip install dvc`
- **Docker** (Optional, for containerized deployment) - [Download Docker](https://www.docker.com/get-started)

## 🏗️ Project Structure

```
CommentAnalysis/
├── src/                          # ML Pipeline source code
│   └── CommentAnalysis/
│       ├── components/           # Pipeline components
│       ├── config/               # Configuration management
│       ├── pipeline/             # DVC pipeline stages
│       └── utils/                # Utility functions
├── CommentAnaysisExtension/      # Application deployment
│   ├── backend/                  # Flask API backend
│   │   ├── app.py                # Main Flask application
│   │   ├── config.py             # Configuration management
│   │   ├── requirements.txt      # Python dependencies
│   │   ├── run_local.py          # Local development runner
│   │   └── .env                  # Environment variables (create this)
│   └── youtube-sentiment-extension/  # Chrome extension (frontend)
│       ├── src/                  # Vue.js source code
│       ├── package.json          # Node.js dependencies
│       ├── vite.config.js        # Vite configuration
│       └── dist/                 # Built extension files
├── config/                       # Configuration files
├── research/                     # Jupyter notebooks
├── scripts/                      # Utility scripts
└── artifacts/                    # DVC tracked artifacts
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CommentAnalysis
```

### 2. Setup Python Environment (for Backend & ML Pipeline)

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install ML pipeline dependencies
pip install -r requirements.txt

# Install backend dependencies
cd CommentAnaysisExtension/backend
pip install -r requirements.txt
cd ../..
```

### 3. Setup Node.js Environment (for Frontend)

```bash
cd CommentAnaysisExtension/youtube-sentiment-extension
npm install
cd ../..
```

### 4. Configure Environment Variables

#### Backend Configuration

Create a `.env` file in `CommentAnaysisExtension/backend/`:

```bash
cd CommentAnaysisExtension/backend
# Create .env file
# On Windows (PowerShell):
New-Item -ItemType File -Path .env
# On macOS/Linux:
touch .env
```

Add the following to `.env`:

```env
# Dagshub Credentials (Required for model loading)
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token

# Flask Configuration
FLASK_ENV=development
PORT=8000
HOST=127.0.0.1
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080

# MLflow Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/AIwithAj/CommentAnalysis.mlflow
MLFLOW_MODEL_NAME=yt_chrome_plugin_model
MLFLOW_VECTORIZER_RUN_ID=e7456a00a6d74d1f9dfc2da425a41d24
MLFLOW_ARTIFACT_PATH=transformer.pkl

# Logging
LOG_LEVEL=DEBUG
```

**Note**: Replace `your_username` and `your_token` with your actual Dagshub credentials.

## 🔧 Running the Backend

The backend is a Flask REST API that provides sentiment analysis endpoints.

### Option 1: Using Python Script (Recommended for Development)

```bash
# Navigate to backend directory
cd CommentAnaysisExtension/backend

# Make sure virtual environment is activated
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run the backend
python run_local.py
```

The backend will start on `http://127.0.0.1:8000`

### Option 2: Using Flask Directly

```bash
cd CommentAnaysisExtension/backend

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Run Flask app
python app.py
```

### Option 3: Using Docker

```bash
# From project root
docker-compose up backend

# Or build and run manually
cd CommentAnaysisExtension/backend
docker build -t comment-analysis-backend .
docker run -p 8000:8000 --env-file .env comment-analysis-backend
```

### Backend API Endpoints

Once the backend is running, you can access:

- **Health Check**: `http://127.0.0.1:8000/health`
- **Readiness Check**: `http://127.0.0.1:8000/ready`
- **Demo Analysis**: `http://127.0.0.1:8000/demo` (GET)
- **Analyze Comments**: `http://127.0.0.1:8000/analyze_comments` (POST)
- **Predict with Timestamps**: `http://127.0.0.1:8000/predict_with_timestamps` (POST)

### Testing the Backend

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test demo endpoint
curl http://127.0.0.1:8000/demo

# Test analyze endpoint
curl -X POST http://127.0.0.1:8000/analyze_comments \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      {
        "text": "This video is amazing!",
        "timestamp": "2025-01-15T10:00:00Z",
        "authorId": "user1"
      }
    ]
  }'
```

## 🎨 Running the Frontend

The frontend is a Chrome extension built with Vue.js and Vite.

### Development Mode

```bash
# Navigate to frontend directory
cd CommentAnaysisExtension/youtube-sentiment-extension

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

This will start the Vite development server. However, for a Chrome extension, you'll need to build it first.

### Building the Extension

```bash
cd CommentAnaysisExtension/youtube-sentiment-extension

# Build the extension
npm run build
```

This creates a `dist/` folder with the built extension files.

### Loading the Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `CommentAnaysisExtension/youtube-sentiment-extension/dist` folder
5. The extension should now be loaded and ready to use

### Frontend Development Workflow

1. Make changes to files in `src/`
2. Run `npm run build` to rebuild
3. In Chrome extensions page, click the reload icon on your extension
4. Test your changes

## 🚀 Running the Complete Project

To run both backend and frontend together:

### Terminal 1 - Backend

```bash
# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Navigate to backend
cd CommentAnaysisExtension/backend

# Run backend
python run_local.py
```

### Terminal 2 - Frontend (if needed for development)

```bash
cd CommentAnaysisExtension/youtube-sentiment-extension

# Build extension
npm run build

# Or run dev server (for testing components)
npm run dev
```

### Using Docker Compose (All Services)

```bash
# From project root
docker-compose up
```

This will start the backend service. The frontend extension needs to be built and loaded manually in Chrome.

## 📊 ML Pipeline

The ML pipeline uses DVC for orchestration:

```bash
# View pipeline structure
dvc dag

# Run complete pipeline
dvc repro

# Run specific stage
dvc repro stage_01_data_ingestion

# Pull data artifacts
dvc pull

# Push artifacts
dvc push
```

### Pipeline Stages

1. **Data Ingestion** (`stage_01_data_ingestion`) - Download and store data
2. **Data Validation** (`stage_02_data_validation`) - Validate and split data
3. **Data Transformation** (`stage_03_data_transformation`) - Feature engineering
4. **Model Training** (`stage_04_model_trainer`) - Train ML model
5. **Model Evaluation** (`stage_05_Evaluation`) - Evaluate and register model

## 🔧 Development

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Run tests
make test

# All checks
make check-all
```

### Backend Testing

```bash
cd CommentAnaysisExtension/backend

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🐳 Docker Deployment

### Backend Only

```bash
# Build image
cd CommentAnaysisExtension/backend
docker build -t comment-analysis-backend .

# Run container
docker run -p 8000:8000 --env-file .env comment-analysis-backend
```

### Using Docker Compose

```bash
# From project root
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🐛 Troubleshooting

### Backend Issues

#### Model Not Loading
- **Problem**: Health check returns "degraded" status
- **Solution**: 
  - Verify `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are set in `.env`
  - Check internet connection (model is downloaded from Dagshub)
  - Verify MLflow tracking URI is correct

#### Port Already in Use
- **Problem**: `Address already in use` error
- **Solution**: 
  - Change `PORT` in `.env` to a different port (e.g., 8001)
  - Or stop the process using port 8000

#### Import Errors
- **Problem**: Module not found errors
- **Solution**: 
  - Ensure virtual environment is activated
  - Reinstall dependencies: `pip install -r requirements.txt`

### Frontend Issues

#### Build Errors
- **Problem**: `npm run build` fails
- **Solution**: 
  - Delete `node_modules` and `package-lock.json`
  - Run `npm install` again
  - Check Node.js version (should be 16+)

#### Extension Not Loading
- **Problem**: Chrome shows errors when loading extension
- **Solution**: 
  - Ensure you're loading the `dist/` folder, not `src/`
  - Check browser console for errors
  - Verify `manifest.json` is in the `dist/` folder

#### CORS Errors
- **Problem**: Frontend can't connect to backend
- **Solution**: 
  - Verify backend is running
  - Check `ALLOWED_ORIGINS` in backend `.env` includes your frontend URL
  - Ensure backend CORS is properly configured

### DVC Issues

```bash
# Reinitialize DVC
dvc init

# Check status
dvc status

# Pull missing files
dvc pull

# Remove cache and re-run
dvc cache dir
dvc repro --force
```

## 📚 Additional Documentation

- [Backend README](CommentAnaysisExtension/backend/README.md) - Detailed backend API documentation
- [AWS Deployment Guide](CommentAnaysisExtension/backend/AWS_DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [DVC Documentation](https://dvc.org/doc) - Data version control guide

## 🔒 Security

- ✅ Secrets stored in environment variables
- ✅ Rate limiting on API endpoints
- ✅ Input validation and sanitization
- ✅ Security scanning in CI/CD
- ✅ Non-root Docker containers

## 📊 Monitoring

- Structured logging
- CloudWatch integration (production)
- Health check endpoints (`/health`, `/ready`)
- Request/response logging

## 🤝 Contributing

1. Install pre-commit hooks: `pre-commit install`
2. Make your changes
3. Run tests: `make test`
4. Format code: `make format`
5. Commit (pre-commit will run automatically)

## 📝 Environment Variables Summary

### Backend Required Variables
- `DAGSHUB_USERNAME` - Your Dagshub username
- `DAGSHUB_TOKEN` - Your Dagshub access token

### Backend Optional Variables
- `FLASK_ENV` - Environment (development/production)
- `PORT` - Server port (default: 8000)
- `HOST` - Server host (default: 127.0.0.1)
- `ALLOWED_ORIGINS` - CORS allowed origins
- `LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)

## 📄 License

See LICENSE file.

## 🙏 Acknowledgments

- DVC for data versioning
- MLflow for model tracking
- Dagshub for MLflow hosting
- Flask for API framework
- Vue.js for frontend framework
- Vite for build tooling

---

**Status**: ✅ Production Ready

**Last Updated**: 2025-01-15
