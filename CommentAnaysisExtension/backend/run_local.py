"""
Simple script to run the backend locally for testing.
This handles environment setup and runs the Flask app.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
else:
    # Try loading from parent directory as well
    parent_env_path = Path(__file__).parent.parent / '.env'
    if parent_env_path.exists():
        load_dotenv(parent_env_path)
        print(f"✅ Loaded environment variables from {parent_env_path}")
    else:
        print("⚠️  No .env file found. Using system environment variables only.")

# Set default environment variables for local development if not set
if not os.getenv("FLASK_ENV"):
    os.environ["FLASK_ENV"] = "development"
if not os.getenv("PORT"):
    os.environ["PORT"] = "8000"
if not os.getenv("HOST"):
    os.environ["HOST"] = "127.0.0.1"
if not os.getenv("ALLOWED_ORIGINS"):
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080"
if not os.getenv("LOG_LEVEL"):
    os.environ["LOG_LEVEL"] = "DEBUG"

# Check for required credentials
if not os.getenv("DAGSHUB_USERNAME") or not os.getenv("DAGSHUB_TOKEN"):
    print("⚠️  WARNING: DAGSHUB_USERNAME and DAGSHUB_TOKEN not set!")
    print("   The app will start but model loading will fail.")
    print("   Set these environment variables to load the model:")
    print("   - DAGSHUB_USERNAME")
    print("   - DAGSHUB_TOKEN")
    print("\n   You can still test endpoints that don't require the model.")
    print("   Continuing anyway...\n")

# Import and run the app
try:
    from app import app, config
    
    print("=" * 60)
    print("🚀 Starting Comment Analysis Backend")
    print("=" * 60)
    print(f"Environment: {config.FLASK_ENV}")
    print(f"Host: {config.HOST}")
    print(f"Port: {config.PORT}")
    print(f"Debug: {config.DEBUG}")
    print("=" * 60)
    print("\n📡 API Endpoints:")
    print(f"   - http://{config.HOST}:{config.PORT}/")
    print(f"   - http://{config.HOST}:{config.PORT}/health")
    print(f"   - http://{config.HOST}:{config.PORT}/ready")
    print(f"   - http://{config.HOST}:{config.PORT}/demo")
    print(f"   - http://{config.HOST}:{config.PORT}/analyze_comments")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
except KeyboardInterrupt:
    print("\n\n👋 Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

