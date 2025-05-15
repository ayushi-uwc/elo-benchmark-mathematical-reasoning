# Has all the configuration for the tournament

import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Get MongoDB connection string
MONGODB_URI = os.getenv("MONGODB_URI")

# Tournament settings
SWISS_DELTA = 100  # maximum Elo difference allowed in a pair
JUDGE_COUNT = 3    # number of judge models
ELO_TEMP = 300     # Softmax temperature for vote weighting
K_FACTOR = 16      # ELO K-factor for rating updates
MAX_MATCHES = 20   # Default max matches for tournament batch

# Database configuration
DB_NAME = "llm_tournament"
MODELS_COLLECTION = "models"
MATCHES_COLLECTION = "matches"

# Set environment variables for litellm to use
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY