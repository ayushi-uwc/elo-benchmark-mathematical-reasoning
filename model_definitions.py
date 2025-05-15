# model_definitions.py
"""
Definitions of LLM models used in the tournament.
This file can be updated independently of the model class implementation.
"""

# List of models with their properties
MODELS = [
    {
        "name": "GPT-4o", 
        "model_id": "openai/gpt-4o", 
        "provider": "openai", 
        "param_count": 1000, 
        "context_window": 128000, 
        "input_cost_per_million": 5.00,
        "output_cost_per_million": 15.00,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "GPT-o4-mini", 
        "model_id": "openai/o4-mini-2025-04-16", 
        "provider": "openai", 
        "param_count": 1000, 
        "context_window": 128000, 
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 7.50,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "GPT-o3-mini", 
        "model_id": "openai/o3-mini-2025-01-31", 
        "provider": "openai", 
        "param_count": 1000, 
        "context_window": 128000, 
        "input_cost_per_million": 1.10,
        "output_cost_per_million": 4.40,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "Claude 3.5 Sonnet", 
        "model_id": "anthropic/claude-3-5-sonnet-20240620", 
        "provider": "anthropic", 
        "param_count": 2000, 
        "context_window": 200000, 
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "Claude 3 Sonnet", 
        "model_id": "anthropic/claude-3-sonnet-20240229", 
        "provider": "anthropic", 
        "param_count": 2000, 
        "context_window": 200000, 
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "LLaMA 3.1 8B Instant", 
        "model_id": "groq/llama-3.1-8b-instant", 
        "provider": "groq", 
        "param_count": 8, 
        "context_window": 8192, 
        "input_cost_per_million": 0.10,
        "output_cost_per_million": 0.40,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "Gemma 2 9B", 
        "model_id": "groq/gemma2-9b-it", 
        "provider": "groq", 
        "param_count": 9, 
        "context_window": 8192, 
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.20,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "Llama 3.3 70B", 
        "model_id": "groq/llama-3.3-70b-versatile", 
        "provider": "groq", 
        "param_count": 70, 
        "context_window": 8192, 
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "DeepSeek R1 70B", 
        "model_id": "groq/deepseek-r1-distill-llama-70b", 
        "provider": "groq", 
        "param_count": 70, 
        "context_window": 8192, 
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.20,
        "is_proprietary": True,
        "model_config": None  # No additional config needed
    },
    {
        "name": "Qwen", 
        "model_id": "huggingface/Qwen/Qwen3-235B-A22B", 
        "provider": "huggingface", 
        "param_count": 235, 
        "context_window": 40960,  # From model config
        "input_cost_per_million": 0.025,
        "output_cost_per_million": 0.05,
        "is_proprietary": False,
        "model_config": {
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
]

# Additional model configurations can be added here if needed 

MODEL_CAPS = {
    "max_matches_per_model": 1,      # Maximum matches a model can play in total
}

# For backward compatibility with code that expects cost_per_million
for model in MODELS:
    if "cost_per_million" not in model:
        model["cost_per_million"] = model["output_cost_per_million"]