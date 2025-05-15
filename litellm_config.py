# litellm_config.py
"""
Configure LiteLLM to use local model configurations instead of making API calls.
Import this module before using litellm.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import ModelResponse
    from litellm.llms.huggingface import HuggingfaceConfig
    has_litellm = True
except ImportError:
    logger.warning("LiteLLM is not installed")
    has_litellm = False

# Dictionary of model configurations to avoid API calls
# This will be populated from the model_definitions file
MODEL_CONFIGS = {}

def register_model_config(model_id: str, config: Dict[str, Any]):
    """
    Register a model configuration with LiteLLM.
    
    Args:
        model_id: The model identifier (e.g., 'Qwen/Qwen3-235B-A22B')
        config: The model configuration
    """
    global MODEL_CONFIGS
    
    if not has_litellm:
        logger.warning("LiteLLM is not installed, cannot register model config")
        return
    
    # Add to our configuration cache
    MODEL_CONFIGS[model_id] = config
    logger.info(f"Registered model configuration for {model_id}")
    
    # Create a custom model callback that provides the configuration
    def model_info_callback(model_name, custom_llm_provider, *args, **kwargs):
        """Custom callback to provide model configuration."""
        if model_name == model_id or model_name.endswith(f"/{model_id}"):
            logger.info(f"Using cached model configuration for {model_id}")
            return config
        return None
    
    # Register the callback with LiteLLM
    if hasattr(litellm, "register_model_info_callback"):
        litellm.register_model_info_callback(model_id, model_info_callback)
        logger.info(f"Registered model info callback for {model_id}")

def init_litellm_config():
    """Initialize LiteLLM configuration."""
    if not has_litellm:
        logger.warning("LiteLLM is not installed, skipping configuration")
        return
    
    # Set up logging
    litellm.set_verbose = False  # Reduce verbosity

    # Register the models from model_definitions
    try:
        from model_definitions import MODELS
        for model in MODELS:
            if model["provider"] == "huggingface" and model["model_config"]:
                model_id = model["model_id"]
                # Remove huggingface/ prefix if present
                if model_id.startswith("huggingface/"):
                    model_id = model_id[len("huggingface/"):]
                
                register_model_config(model_id, model["model_config"])
        
        logger.info(f"Registered {len(MODEL_CONFIGS)} model configurations with LiteLLM")
    except Exception as e:
        logger.error(f"Error registering model configurations: {str(e)}")

# Initialize when imported
init_litellm_config() 