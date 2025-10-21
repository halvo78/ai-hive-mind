
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_hive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('AIHive')

class AILogger:
    """Centralized logging for AI Hive"""
    
    @staticmethod
    def info(message):
        logger.info(message)
    
    @staticmethod
    def error(message, exc_info=None):
        logger.error(message, exc_info=exc_info)
    
    @staticmethod
    def warning(message):
        logger.warning(message)
    
    @staticmethod
    def debug(message):
        logger.debug(message)
    
    @staticmethod
    def log_query(prompt, source, cost, latency, success):
        """Log AI query details"""
        logger.info(f"Query: prompt_len={len(prompt)}, source={source}, "
                   f"cost=${cost:.6f}, latency={latency:.2f}s, success={success}")
    
    @staticmethod
    def log_error(operation, error, context=None):
        """Log error with context"""
        logger.error(f"Error in {operation}: {error}", exc_info=True)
        if context:
            logger.error(f"Context: {context}")
