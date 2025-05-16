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

# Dictionary for hard-coded model configurations
# This is only needed for complex models like Hugging Face transformers
# For most API-based models (OpenAI, Anthropic, etc.), this isn't necessary
HARD_CODED_CONFIGS = {
    "Qwen/Qwen3-235B-A22B": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 40960,
        "max_window_layers": 94,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 1536,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 94,
        "num_key_value_heads": 4,
        "output_router_logits": False,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936
    }
}

def register_model_config(model_id: str, config: Dict[str, Any] = None):
    """
    Register a model configuration with LiteLLM.
    
    Args:
        model_id: The model identifier (e.g., 'Qwen/Qwen3-235B-A22B')
        config: The model configuration (optional, will use hard-coded config if available)
    """
    if not has_litellm:
        logger.warning("LiteLLM is not installed, cannot register model config")
        return
    
    # If no config provided, check for hard-coded config
    if config is None:
        # Remove huggingface/ prefix if present for lookup
        lookup_id = model_id
        if lookup_id.startswith("huggingface/"):
            lookup_id = lookup_id[len("huggingface/"):]
            
        config = HARD_CODED_CONFIGS.get(lookup_id)
        if not config:
            logger.warning(f"No configuration available for {model_id}")
            return
    
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

    # Register the Hugging Face models that need configurations
    for model_id, config in HARD_CODED_CONFIGS.items():
        register_model_config(model_id, config)
        
    logger.info(f"Registered {len(HARD_CODED_CONFIGS)} model configurations with LiteLLM")

# Initialize when imported
init_litellm_config() 