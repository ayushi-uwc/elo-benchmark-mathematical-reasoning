# 
import time
import json
import asyncio
import re  # Add regex import
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from config import DB_NAME, MODELS_COLLECTION
from database import db

# Import litellm configuration to use local model configs
import litellm_config

# Now import litellm
from litellm import completion
import logging

# Import new Google GenAI SDK for Gemini models
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    logger = logging.getLogger(__name__)
    logger.warning("google-genai package not installed. Gemini models will use fallback LiteLLM.")

# Import model definitions from separate file
from model_definitions import MODELS

logger = logging.getLogger(__name__)

class LLMModel:
    """
    Class representing an LLM with ELO rating and performance metrics.
    """
    def __init__(
        self, 
        name: str, 
        model_id: str, 
        provider: str,
        pricing_source: str,
        input_cost_per_million: float,
        output_cost_per_million: float,
        max_matches: int = 50,
        api_key: Optional[str] = None,
        initialize_new: bool = True
    ):
        logger.info(f"Initializing model: {name}")
        logger.info(f"Provider: {provider}")
        
        # Basic model info
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.pricing_source = pricing_source
        
        self.api_key = api_key
        self.max_matches = max_matches

        if initialize_new:
            logger.info("Initializing new model with default values")
            # ELO ratings
            self.elo = {
                "raw": {
                    "initial": 1500,
                    "current": 1500
                },
                "cost_adjusted": {
                    "initial": 1500,
                    "current": 1500
                }
            }

            # Performance metrics with properly initialized nested structures
            self.performance = {
                "total_matches_played": 0,
                "wins_raw": 0,  # Individual judge votes for win
                "losses_raw": 0,  # Individual judge votes for loss
                "draws_raw": 0,  # Individual judge votes for draw
                "total_tokens_used": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "score_history": {
                    "raw_scores": [],
                    "adjusted_scores": [],
                    "avg_raw_score": 0.0,
                    "avg_adjusted_score": 0.0
                }
            }

            # Match tracking
            self.match_ids = {
                "played": [],
                "judged": [],
                "cases_generated": [],
                "questions_generated": []
            }

            # Metadata
            self.metadata = {
                "notes": "",
                "last_updated": datetime.utcnow().isoformat() + "Z"
            }
        
    def generate(self, prompt):
        """Generate a response using the LLM through litellm or native Gemini API."""
        logger.info(f"Generating response with {self.name}")
        start = time.time()

        try:
            # Use native Google GenAI SDK for Gemini models
            if self.provider == "gemini" and HAS_GOOGLE_GENAI:
                return self._generate_with_gemini(prompt, start)
            else:
                # Use LiteLLM for all other models
                return self._generate_with_litellm(prompt, start)
                
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            raise e

    def _generate_with_gemini(self, prompt, start_time):
        """Generate response using native Google GenAI SDK."""
        try:
            # Initialize Gemini client
            client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            )

            # Extract model name from model_id (remove gemini/ prefix if present)
            model_name = self.model_id.replace("gemini/", "")
            
            # Create content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Generate content
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
            )
            
            # Collect the full response
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
            
            # Filter out any content between XML-style tags
            filtered_response = re.sub(r'<(\w+)>.*?</\1>', '', full_response, flags=re.DOTALL)
            
            # For Gemini, we'll estimate token usage since the new API might not provide exact counts
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimation
            completion_tokens = len(full_response.split()) * 1.3
            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost using separate input/output rates
            input_cost = (prompt_tokens / 1000000) * self.input_cost_per_million
            output_cost = (completion_tokens / 1000000) * self.output_cost_per_million
            total_cost = input_cost + output_cost

            # Update model statistics (check if performance exists)
            if hasattr(self, 'performance'):
                self.performance["total_tokens_used"] += total_tokens
                self.performance["total_input_tokens"] += prompt_tokens
                self.performance["total_output_tokens"] += completion_tokens
                self.performance["total_cost_usd"] += total_cost
                
                # Initialize call tracking if missing
                if "total_calls" not in self.performance:
                    self.performance["total_calls"] = 0
                if "total_time" not in self.performance:
                    self.performance["total_time"] = 0
                    
                self.performance["total_calls"] += 1

                # Update timing
                elapsed = time.time() - start_time
                self.performance["total_time"] += elapsed
                self.performance["avg_time_per_call"] = self.performance["total_time"] / self.performance["total_calls"]
            else:
                elapsed = time.time() - start_time

            logger.info(f"Generated Gemini response: {len(filtered_response)} chars, {total_tokens} tokens, ${total_cost:.4f}")

            return {
                "response": filtered_response,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cost": total_cost,
                "latency": elapsed * 1000  # Convert to milliseconds to match LiteLLM format
            }
            
        except Exception as e:
            logger.error(f"Error with Gemini API for {self.name}: {e}")
            # Fallback to LiteLLM if Gemini fails
            logger.info(f"Falling back to LiteLLM for {self.name}")
            return self._generate_with_litellm(prompt, start_time)

    def _generate_with_litellm(self, prompt, start_time):
        """Generate response using LiteLLM (original implementation)."""
        # Make the API call with the model configuration
        response = completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )

        # Extract metrics
        model_response = response['choices'][0]['message']['content']
        
        # Filter out any content between XML-style tags <anyword>...</anyword>
        filtered_response = re.sub(r'<(\w+)>.*?</\1>', '', model_response, flags=re.DOTALL)
        
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']

        # Calculate cost using separate input/output rates
        input_cost = (prompt_tokens / 1000000) * self.input_cost_per_million
        output_cost = (completion_tokens / 1000000) * self.output_cost_per_million
        total_cost = input_cost + output_cost

        # Update model statistics (check if performance exists)
        if hasattr(self, 'performance'):
            self.performance["total_tokens_used"] += total_tokens
            self.performance["total_input_tokens"] += prompt_tokens
            self.performance["total_output_tokens"] += completion_tokens
            self.performance["total_cost_usd"] += total_cost

        end = time.time()
        latency = (end - start_time) * 1000  # ms

        logger.info(f"Generation complete - Input tokens: {prompt_tokens}, Output tokens: {completion_tokens}")
        logger.info(f"Costs: Input: ${input_cost:.4f}, Output: ${output_cost:.4f}, Total: ${total_cost:.4f}, Latency: {latency:.2f}ms")

        # Update metadata (check if metadata exists)
        if hasattr(self, 'metadata'):
            self.metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
            self.save_to_db()

        return {
            "response": filtered_response,  # Return filtered response
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cost": total_cost,
            "latency": latency
        }
    
    async def async_generate(self, prompt):
        """Async version of generate method for concurrent processing."""
        # Use an executor to run the synchronous generate method in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.generate, prompt)
        return result
    
    @property
    def avg_tokens_per_response(self):
        """Calculate average tokens per response."""
        if self.performance["total_matches_played"] > 0:
            return self.performance["total_tokens_used"] / self.performance["total_matches_played"]
        return 0
    
    @property
    def avg_cost_per_response_usd(self):
        """Calculate average cost per response in USD."""
        if self.performance["total_matches_played"] > 0:
            return self.performance["total_cost_usd"] / self.performance["total_matches_played"]
        return 0
            
    def to_dict(self):
        """Convert model object to dictionary for MongoDB storage."""
        model_dict = {
            "name": self.name,
            "model_id": self.model_id,
            "provider": self.provider,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "cost_per_million": self.output_cost_per_million,  # For backward compatibility
            "pricing_source": self.pricing_source,
            "api_key": self.api_key,
            "max_matches": self.max_matches,
            "elo": self.elo,
            "performance": self.performance,
            "match_ids": self.match_ids,
            "metadata": self.metadata
        }
        
        # Add calculated fields
        model_dict["performance"]["avg_tokens_per_response"] = self.avg_tokens_per_response
        model_dict["performance"]["avg_cost_per_response_usd"] = self.avg_cost_per_response_usd
        
        return model_dict
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model object from dictionary (from MongoDB)."""
        # Handle both old and new cost formats
        input_cost = data.get("input_cost_per_million", data.get("cost_per_million", 0))
        output_cost = data.get("output_cost_per_million", data.get("cost_per_million", 0))
        
        # Create model with basic attributes
        model = cls(
            name=data["name"],
            model_id=data["model_id"],
            provider=data["provider"],
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
            max_matches=data.get("max_matches", 50),
            pricing_source=data.get("pricing_source", ""),
            api_key=data.get("api_key"),
            initialize_new=False  # Don't initialize new values
        )
        
        # Set complex attributes
        model.elo = data.get("elo", {
            "raw": {"initial": 1500, "current": 1500},
            "cost_adjusted": {"initial": 1500, "current": 1500}
        })
        
        # Get performance data with proper defaults for all fields
        default_performance = {
            "total_matches_played": 0,
            "wins_raw": 0,
            "losses_raw": 0,
            "wins_adjusted": 0,
            "losses_adjusted": 0,
            "draws_raw": 0,
            "total_tokens_used": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "judge_votes": {
                "received_wins": 0,
                "received_losses": 0,
                "received_draws": 0,
                "given_wins": 0,
                "given_losses": 0,
                "given_draws": 0
            },
            "score_history": {
                "raw_scores": [],
                "adjusted_scores": [],
                "avg_raw_score": 0.0,
                "avg_adjusted_score": 0.0
            }
        }
        
        # Update with actual data, maintaining structure
        performance_data = data.get("performance", {})
        for key, value in performance_data.items():
            if key in default_performance:
                if isinstance(default_performance[key], dict):
                    # Merge nested dictionaries
                    default_performance[key].update(value)
                else:
                    default_performance[key] = value
                    
        model.performance = default_performance
        
        model.match_ids = data.get("match_ids", {
            "played": [],
            "judged": [],
            "cases_generated": [],
            "questions_generated": []
        })
        
        model.metadata = data.get("metadata", {
            "notes": "",
            "last_updated": datetime.utcnow().isoformat() + "Z"
        })
                
        return model
    
    def update_elo(self, opponent, match_result, match_id, raw_score, adjusted_score=None):
        """
        Update ELO ratings based on match results.
        
        Args:
            opponent: The opponent LLMModel
            match_result: String 'win', 'loss', or 'draw'
            match_id: Unique ID for the match
            raw_score: Raw performance score (0-1)
            adjusted_score: Cost-adjusted score (0-1), if None uses raw_score
        """
        logger.info(f"Updating ELO for {self.name} vs {opponent.name}")
        logger.info(f"Match result: {match_result}, Raw score: {raw_score:.2f}")
        
        if adjusted_score is None:
            adjusted_score = raw_score
            
        # Update raw ELO
        K = 32  # K-factor determines how much ratings change
        
        # Expected scores based on current ratings
        expected_raw = 1.0 / (1.0 + 10.0 ** ((opponent.elo["raw"]["current"] - self.elo["raw"]["current"]) / 400.0))
        expected_adj = 1.0 / (1.0 + 10.0 ** ((opponent.elo["cost_adjusted"]["current"] - self.elo["cost_adjusted"]["current"]) / 400.0))
        
        # Update ELO ratings
        old_elo_raw = self.elo["raw"]["current"]
        old_elo_adj = self.elo["cost_adjusted"]["current"]
        
        self.elo["raw"]["current"] += K * (raw_score - expected_raw)
        self.elo["cost_adjusted"]["current"] += K * (adjusted_score - expected_adj)
        
        logger.info(f"ELO change - Raw: {old_elo_raw:.1f} -> {self.elo['raw']['current']:.1f}")
        logger.info(f"ELO change - Adjusted: {old_elo_adj:.1f} -> {self.elo['cost_adjusted']['current']:.1f}")
        
        # Update performance metrics
        self.performance["total_matches_played"] += 1
        self.match_ids["played"].append(match_id)
        
        if match_result == "win":
            self.performance["wins_raw"] += 1
            if adjusted_score > 0.5:
                self.performance["wins_adjusted"] += 1
            else:
                self.performance["losses_adjusted"] += 1
        elif match_result == "loss":
            self.performance["losses_raw"] += 1
            if adjusted_score > 0.5:
                self.performance["wins_adjusted"] += 1
            else:
                self.performance["losses_adjusted"] += 1
        
        logger.info(f"Updated record: {self.performance['wins_raw']}-{self.performance['losses_raw']}")
        
        # Update metadata
        self.metadata["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Save to database
        self.save_to_db()
    
    def save_to_db(self):
        """Save model to MongoDB."""
        return db.save_model(self.to_dict())
    
    @classmethod
    def load_from_db(cls, name):
        """Load model from MongoDB by name."""
        model_data = db.get_model(name)
        if model_data:
            return cls.from_dict(model_data)
        return None
    
    def __str__(self):
        return f"{self.name} (ELO: {self.elo['raw']['current']:.1f}, Cost-Adj: {self.elo['cost_adjusted']['current']:.1f}, Matches: {self.performance['total_matches_played']})"

    def __repr__(self):
        return self.__str__()


def initialize_models(models_data: List[Dict[str, Any]]):
    """
    Initialize models from the provided list.
    If a model exists in the database, load it; otherwise create and save it.
    
    Args:
        models_data: List of dictionaries with model information
        
    Returns:
        List of LLMModel objects
    """
    initialized_models = []
    
    for model_info in models_data:
        # Check if model exists in database
        existing_model = LLMModel.load_from_db(model_info["name"])
        
        if existing_model:
            print(f"Loaded existing model: {existing_model}")
            initialized_models.append(existing_model)
        else:
            # Create new model
            new_model = LLMModel(
                name=model_info["name"],
                model_id=model_info["model_id"],
                provider=model_info["provider"],
                input_cost_per_million=model_info.get("input_cost_per_million", model_info.get("cost_per_million", 0)),
                output_cost_per_million=model_info.get("output_cost_per_million", model_info.get("cost_per_million", 0)),
                max_matches=model_info.get("max_matches", 50),
                pricing_source=model_info.get("pricing_source", ""),
                api_key=model_info.get("api_key")
            )
            
            # Save to database
            new_model.save_to_db()
            print(f"Created new model: {new_model}")
            initialized_models.append(new_model)
    
    print(f"Initialized {len(initialized_models)} models")
    return initialized_models


# This should be at the end of the file
if __name__ == "__main__":
    # Import models from the model_definitions file
    from model_definitions import MODELS
    
    # Initialize models
    all_models = initialize_models(MODELS)
    
    # Print models
    for model in all_models:
        print(model)