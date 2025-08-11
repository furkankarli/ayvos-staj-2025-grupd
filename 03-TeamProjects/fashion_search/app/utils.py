import os

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Config:
    """Configuration class"""

    # App Settings
    APP_NAME = os.getenv("APP_NAME", "Fashion Search API")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Model Settings
    MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")
    MODEL_PRETRAINED = os.getenv("MODEL_PRETRAINED", "laion2b_s34b_b79k")

    # Data Settings
    DATA_PATH = os.getenv("DATA_PATH", "./data/fashion")
    EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./data/embeddings")

    # Log Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")

    # Performance Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


def setup_logger():
    """Configure the Loguru logger"""

    # Remove the default logger
    logger.remove()

    # Add a console logger
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=Config.LOG_LEVEL,
        format=log_format,
        colorize=True,
    )

    # Add a file logger
    file_log_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} - {message}"
    )
    logger.add(
        sink=Config.LOG_FILE,
        level=Config.LOG_LEVEL,
        format=file_log_format,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logger configured for {Config.APP_NAME} v{Config.APP_VERSION}")


# Initialize the logger
setup_logger()

# For testing purposes
if __name__ == "__main__":
    logger.info("Configuration test completed successfully!")
    logger.debug(f"DEBUG mode: {Config.DEBUG}")
    logger.info(f"App will run on {Config.HOST}:{Config.PORT}")
