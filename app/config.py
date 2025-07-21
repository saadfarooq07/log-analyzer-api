"""Configuration management for the Log Analyzer API."""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys (required)
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    
    # Application settings
    app_name: str = "Log Analyzer API"
    app_version: str = "1.0.0"
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # API settings
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["*"]
    
    # Analysis settings
    max_log_size_mb: int = Field(10, env="MAX_LOG_SIZE_MB")
    enable_streaming: bool = Field(True, env="ENABLE_STREAMING")
    analysis_timeout: int = Field(300, env="ANALYSIS_TIMEOUT")  # seconds
    
    # Model settings
    primary_model: str = Field("gemini-1.5-flash", env="PRIMARY_MODEL")
    orchestration_model: str = Field("kimi-k2", env="ORCHESTRATION_MODEL")
    temperature: float = Field(0.7, env="TEMPERATURE")
    max_tokens: int = Field(4096, env="MAX_TOKENS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()