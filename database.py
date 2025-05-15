# Has all the database operations

import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import certifi
import logging
from typing import Dict, List, Any, Optional

# Import configuration
from config import MONGODB_URI, DB_NAME, MODELS_COLLECTION, MATCHES_COLLECTION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection and operations helper class."""
    
    def __init__(self, connection_string: str = MONGODB_URI, db_name: str = DB_NAME):
        """Initialize MongoDB connection with proper error handling."""
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.models = None
        self.matches = None
        self.connected = False
        
        # Attempt connection
        self.connect()
        
    def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            # Initialize with SSL certificate path from certifi
            self.client = MongoClient(
                self.connection_string, 
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000  # 5 seconds timeout
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Configure database and collections
            self.db = self.client[self.db_name]
            self.models = self.db[MODELS_COLLECTION]
            self.matches = self.db[MATCHES_COLLECTION]
            
            # Create indexes for better performance
            self.models.create_index([("name", pymongo.ASCENDING)], unique=True)
            
            self.connected = True
            logger.info(f"Successfully connected to MongoDB: {self.db_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if the database connection is active."""
        if not self.connected or not self.client:
            return False
            
        try:
            # Test connection is still alive
            self.client.admin.command('ping')
            return True
        except:
            self.connected = False
            return False
    
    # Model operations
    def save_model(self, model_data: Dict[str, Any]) -> bool:
        """Save model data to MongoDB."""
        if not self.is_connected():
            logger.error("Cannot save model: Not connected to MongoDB")
            return False
            
        try:
            # Use model name as unique identifier
            query = {"name": model_data["name"]}
            self.models.update_one(query, {"$set": model_data}, upsert=True)
            logger.info(f"Saved model: {model_data['name']}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_data.get('name', 'unknown')}: {str(e)}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by name."""
        if not self.is_connected():
            logger.error("Cannot get model: Not connected to MongoDB")
            return None
            
        try:
            return self.models.find_one({"name": model_name}, {"_id": 0})
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {str(e)}")
            return None
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Retrieve all models from the database."""
        if not self.is_connected():
            logger.error("Cannot get models: Not connected to MongoDB")
            return []
            
        try:
            return list(self.models.find({}, {"_id": 0}))
        except Exception as e:
            logger.error(f"Error getting all models: {str(e)}")
            return []
    
    # Match operations
    def save_match(self, match_data: Dict[str, Any]) -> bool:
        """Save match result to MongoDB."""
        if not self.is_connected():
            logger.error("Cannot save match: Not connected to MongoDB")
            return False
            
        try:
            self.matches.insert_one(match_data)
            logger.info(f"Saved match: {match_data.get('model_a', 'unknown')} vs {match_data.get('model_b', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Error saving match: {str(e)}")
            return False
    
    def get_match(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific match by ID."""
        if not self.is_connected():
            logger.error("Cannot get match: Not connected to MongoDB")
            return None
            
        try:
            return self.matches.find_one({"_id": match_id})
        except Exception as e:
            logger.error(f"Error getting match {match_id}: {str(e)}")
            return None
    
    def get_matches(self, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve matches with optional filtering."""
        if not self.is_connected():
            logger.error("Cannot get matches: Not connected to MongoDB")
            return []
            
        if query is None:
            query = {}
            
        try:
            return list(self.matches.find(query, {"_id": 0}))
        except Exception as e:
            logger.error(f"Error getting matches: {str(e)}")
            return []
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB connection closed")

# Create a global database instance
db = MongoDB()

# Function to check if connection succeeded and provide diagnostic info
def check_connection_status():
    """Check and print database connection status."""
    if db.is_connected():
        print("✅ Successfully connected to MongoDB!")
        print(f"  - Database: {db.db_name}")
        print(f"  - Collections: {MODELS_COLLECTION}, {MATCHES_COLLECTION}")
        
        # Print count of existing data if any
        model_count = len(db.get_all_models())
        match_count = len(db.get_matches())
        print(f"  - Existing data: {model_count} models, {match_count} matches")
        return True
    else:
        print("❌ Failed to connect to MongoDB")
        print("  - Check your connection string in the .env file")
        print("  - Ensure your IP address is whitelisted in MongoDB Atlas")
        print("  - Verify that your username and password are correct")
        return False

# Example usage
if __name__ == "__main__":
    # Test the connection
    check_connection_status()