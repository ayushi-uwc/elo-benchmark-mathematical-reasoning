# logger_config.py
import os
import logging
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"tournament_{timestamp}.log")

# Special logger for tables
class TableLogger:
    """A special logger for tables that doesn't add timestamps to log entries."""
    def __init__(self, file_path):
        self.file_path = file_path
        
    def log(self, message):
        """Log a message directly to the file without any prefixes."""
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
                f.flush()
        except Exception as e:
            print(f"Error writing to table log: {str(e)}")

# Configure the root logger
def setup_logger(level=logging.INFO):
    """Set up the root logger with file and console handlers."""
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Set the log level
    root_logger.setLevel(level)
    
    try:
        # Create file handler with immediate flush
        file_handler = logging.FileHandler(log_file, encoding='utf-8', delay=False)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add both handlers to the root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Display absolute path of log file
        abs_path = os.path.abspath(log_file)
        print(f"Log file will be written to: {abs_path}")
        root_logger.info(f"Logging started, writing to: {abs_path}")
        
        # Verify file is writable by testing it directly
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("# Log file initialized\n")
            f.flush()
        
        return root_logger
    
    except Exception as e:
        print(f"ERROR SETTING UP LOGGER: {str(e)}")
        print(f"Attempted to write to: {os.path.abspath(log_file)}")
        print("Falling back to console-only logging")
        
        # If file logging fails, set up console-only logging
        root_logger.addHandler(console_handler)
        return root_logger

# Initialize the root logger
root_logger = setup_logger()

# Create a table logger instance
table_logger = TableLogger(log_file)

# Force a log message to verify logger is working
root_logger.info("Logger initialization complete")

def get_logger(name):
    """Get a logger instance with the given name."""
    logger = logging.getLogger(name)
    
    # Add a function to force flush logs
    def force_flush():
        for handler in root_logger.handlers:
            handler.flush()
    
    logger.force_flush = force_flush
    
    # Add table logging capability
    logger.table = table_logger.log
    
    return logger