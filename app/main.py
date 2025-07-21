"""Main FastAPI application entry point."""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from .api.routes import router
from .config import settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Log Analyzer API...")
    
    # Validate required environment variables
    required_vars = ["GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")
    
    logger.info("✅ Log Analyzer API started successfully")
    
    yield
    
    # Shutdown
    logger.info("👋 Log Analyzer API shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Analyze logs using LangGraph and AI models",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "api": settings.api_prefix
    }


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "healthy",
        "service": "log-analyzer-api",
        "version": settings.app_version
    }


# Ready check endpoint
@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    # Check if all required services are available
    try:
        # You could add checks for external services here
        return {
            "status": "ready",
            "checks": {
                "api_keys": "configured",
                "models": "available"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")