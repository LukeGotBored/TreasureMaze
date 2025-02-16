import logging
import os
import re
from colorama import Fore, Style
from datetime import datetime

LOGS_DIR = "logs"

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    FORMATS = {
        logging.DEBUG: f"{Fore.CYAN}[*]{Style.RESET_ALL} %(message)s",
        logging.INFO: f"{Fore.GREEN}[*]{Style.RESET_ALL} %(message)s",
        logging.WARNING: f"{Fore.YELLOW}[?]{Style.RESET_ALL} %(message)s",
        logging.ERROR: f"{Fore.RED}[!]{Style.RESET_ALL} %(message)s",
        logging.CRITICAL: f"{Fore.RED}{Style.BRIGHT}[!]{Style.RESET_ALL} %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class RemoveColorFilter(logging.Filter):
    ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    def filter(self, record):
        record.msg = self.ANSI_ESCAPE_PATTERN.sub('', str(record.msg))
        return True

def setup_logger(debug: bool = False) -> logging.Logger:
    """Setup logging configuration with both file and console handlers."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"treasure_maze_{timestamp}.log")
        
        logger = logging.getLogger('TreasureMaze')
        # Set level conditionally based on debug flag
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        # Prevent duplicate logging
        if not logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            file_handler.addFilter(RemoveColorFilter())
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(ColoredFormatter())
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    except Exception as e:
        print(f"Failed to setup logger: {e}")
        raise