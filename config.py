"""
Configuration file for RAG API Server
Centralizes application settings and environment variables
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration settings"""

    # Application Settings
    APP_NAME: str = "RAG API Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "False").lower() == "true"

    # CORS Settings
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8080",
        "*"  # Allow all origins - restrict in production
    ]

    # File Storage Settings
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    SUMMARIES_DIR: Path = BASE_DIR / "summaries"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_stores"

    # Document Processing Settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Embedding Model Settings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama2")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Retrieval Settings
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))
    RETRIEVAL_SEARCH_TYPE: str = os.getenv("RETRIEVAL_SEARCH_TYPE", "similarity")

    # Video Analysis Settings
    VIDEO_ANALYSIS_ENABLED: bool = os.getenv(
        "VIDEO_ANALYSIS_ENABLED", "True"
    ).lower() == "true"
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    VIDEO_FRAME_INTERVAL: int = int(os.getenv("VIDEO_FRAME_INTERVAL", "30"))

    # Writer Extraction Settings
    WRITER_EXTRACTION_ENABLED: bool = os.getenv(
        "WRITER_EXTRACTION_ENABLED", "True"
    ).lower() == "true"
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Cache Settings
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.UPLOAD_DIR,
            cls.SUMMARIES_DIR,
            cls.VECTOR_STORE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration"""
        return {
            "app_name": cls.APP_NAME,
            "version": cls.APP_VERSION,
            "host": cls.HOST,
            "port": cls.PORT,
            "debug": cls.DEBUG,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "video_analysis_enabled": cls.VIDEO_ANALYSIS_ENABLED,
            "writer_extraction_enabled": cls.WRITER_EXTRACTION_ENABLED,
        }


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    RELOAD = True


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    RELOAD = False
    CORS_ORIGINS = []  # Should be set explicitly in production


class TestConfig(Config):
    """Test environment configuration"""
    DEBUG = True
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama2"


# Select configuration based on environment
def get_config() -> Config:
    """Get configuration based on environment variable"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig,
    }

    return config_map.get(env, DevelopmentConfig)


# Export default configuration
config = get_config()
