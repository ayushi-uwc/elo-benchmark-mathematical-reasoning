# model_definitions.py
"""
Definitions of LLM models used in the tournament.
This file can be updated independently of the model class implementation.
"""

# List of models with their properties
MODELS = [
  {
    "name": "GPT-4.1 mini",
    "model_id": "openai/gpt-4.1-mini-2025-04-14",
    "provider": "openai",
    "input_cost_per_million": 0.40,
    "output_cost_per_million": 1.60,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-4.1 nano",
    "model_id": "openai/gpt-4.1-nano-2025-04-14",
    "provider": "openai",
    "input_cost_per_million": 0.10,
    "output_cost_per_million": 0.40,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-4o",
    "model_id": "openai/gpt-4o",
    "provider": "openai",
    "input_cost_per_million": 5.00,
    "output_cost_per_million": 20.00,
    "pricing_source": "OpenAI Realtime API pricing, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-4.1",
    "model_id": "openai/gpt-4.1-mini-2025-04-14",
    "provider": "openai",
    "input_cost_per_million": 2.00,
    "output_cost_per_million": 8.00,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-o4-mini",
    "model_id": "openai/o4-mini-2025-04-16",
    "provider": "openai",
    "input_cost_per_million": 1.10,
    "output_cost_per_million": 4.40,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-o3",
    "model_id": "openai/o3-2025-04-16",
    "provider": "openai",
    "input_cost_per_million": 10.00,
    "output_cost_per_million": 40.00,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-3.5 Turbo",
    "model_id": "openai/gpt-3.5-turbo-0125",
    "provider": "openai",
    "input_cost_per_million": 0.50,
    "output_cost_per_million": 1.50,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "GPT-o3-mini",
    "model_id": "openai/o3-mini-2025-01-31",
    "provider": "openai",
    "input_cost_per_million": 10.00,
    "output_cost_per_million": 40.00,
    "pricing_source": "OpenAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3.7 Sonnet",
    "model_id": "anthropic/claude-3-7-sonnet-20250219",
    "provider": "anthropic",
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3.5 Haiku",
    "model_id": "anthropic/claude-3-5-haiku-20241022",
    "provider": "anthropic",
    "input_cost_per_million": 0.80,
    "output_cost_per_million": 4.00,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3.5 Sonnet",
    "model_id": "anthropic/claude-3-5-sonnet-20240620",
    "provider": "anthropic",
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3 Sonnet",
    "model_id": "anthropic/claude-3-sonnet-20240229",
    "provider": "anthropic",
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3 Opus",
    "model_id": "anthropic/claude-3-opus-20240229",
    "provider": "anthropic",
    "input_cost_per_million": 15.00,
    "output_cost_per_million": 75.00,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Claude 3 Haiku",
    "model_id": "anthropic/claude-3-haiku-20240307",
    "provider": "anthropic",
    "input_cost_per_million": 0.25,
    "output_cost_per_million": 1.25,
    "pricing_source": "Anthropic API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "LLaMA 3.1 8B Instant",
    "model_id": "groq/llama-3.1-8b-instant",
    "provider": "groq",
    "input_cost_per_million": 0.05,
    "output_cost_per_million": 0.08,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemma 2 9B",
    "model_id": "groq/gemma2-9b-it",
    "provider": "groq",
    "input_cost_per_million": 0.20,
    "output_cost_per_million": 0.20,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Llama 3.3 70B",
    "model_id": "groq/llama-3.3-70b-versatile",
    "provider": "groq",
    "input_cost_per_million": 0.59,
    "output_cost_per_million": 0.79,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "DeepSeek R1 Distill Llama 70B",
    "model_id": "groq/deepseek-r1-distill-llama-70b",
    "provider": "groq",
    "input_cost_per_million": 0.75,
    "output_cost_per_million": 0.99,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Allamanda 2 7B",
    "model_id": "groq/allam-2-7b",
    "provider": "groq",
    "input_cost_per_million": 0.05,
    "output_cost_per_million": 0.20,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Meta LLama 4 Maverick Instruct 17B",
    "model_id": "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    "provider": "groq",
    "input_cost_per_million": 0.05,
    "output_cost_per_million": 0.20,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Meta LLama 4 Scout Instruct 17B",
    "model_id": "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "provider": "groq",
    "input_cost_per_million": 0.05,
    "output_cost_per_million": 0.20,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Grok 3 Fast",
    "model_id": "xai/grok-3-fast-beta",
    "provider": "xai",
    "input_cost_per_million": 5.00,
    "output_cost_per_million": 25.00,
    "pricing_source": "xAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Grok 3 Mini",
    "model_id": "xai/grok-3-mini-beta",
    "provider": "xai",
    "input_cost_per_million": 0.30,
    "output_cost_per_million": 0.50,
    "pricing_source": "xAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Grok 3",
    "model_id": "xai/grok-3-beta",
    "provider": "xai",
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00,
    "pricing_source": "xAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Grok 3 Mini Fast",
    "model_id": "xai/grok-3-mini-fast-beta",
    "provider": "xai",
    "input_cost_per_million": 0.60,
    "output_cost_per_million": 4.00,
    "pricing_source": "xAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Grok 2",
    "model_id": "xai/grok-2-1212",
    "provider": "xai",
    "input_cost_per_million": 2.00,
    "output_cost_per_million": 10.00,
    "pricing_source": "xAI API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Mistral Saba 24B",
    "model_id": "groq/mistral-saba-24b",
    "provider": "groq",
    "input_cost_per_million": 0.79,
    "output_cost_per_million": 0.79,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 2.5 Pro",
    "model_id": "gemini/gemini-2.5-pro-preview-05-06",
    "provider": "gemini",
    "input_cost_per_million": 1.25,
    "output_cost_per_million": 10.00,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 2.5 Flash",
    "model_id": "gemini/gemini-2.5-flash-preview-04-17",
    "provider": "gemini",
    "input_cost_per_million": 0.15,
    "output_cost_per_million": 0.60,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 2.0 Flash",
    "model_id": "gemini/gemini-2.0-flash",
    "provider": "gemini",
    "input_cost_per_million": 0.10,
    "output_cost_per_million": 0.40,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 2.0 Flash Lite",
    "model_id": "gemini/gemini-2.0-flash-lite",
    "provider": "gemini",
    "input_cost_per_million": 0.075,
    "output_cost_per_million": 0.30,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 1.5 Flash",
    "model_id": "gemini/gemini-1.5-flash",
    "provider": "gemini",
    "input_cost_per_million": 0.075,
    "output_cost_per_million": 0.40,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 1.5 Flash 8B",
    "model_id": "gemini/gemini-1.5-flash-8b",
    "provider": "gemini",
    "input_cost_per_million": 0.0375,
    "output_cost_per_million": 0.15,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemini 1.5 Pro",
    "model_id": "gemini/gemini-1.5-pro",
    "provider": "gemini",
    "input_cost_per_million": 1.25,
    "output_cost_per_million": 5.00,
    "pricing_source": "Gemini API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Gemma 3 1B",
    "model_id": "gemini/gemma-3-1b-it",
    "provider": "gemini",
    "input_cost_per_million": 0.00,
    "output_cost_per_million": 0.00,
    "pricing_source": "Gemini API pricing page, May 2025"
  },
  {
    "name": "Gemma 3 4B",
    "model_id": "gemini/gemma-3-4b-it",
    "provider": "gemini",
    "input_cost_per_million": 0.00,
    "output_cost_per_million": 0.00,
    "pricing_source": "Gemini API pricing page, May 2025"
  },
  {
    "name": "Gemma 3 12B",
    "model_id": "gemini/gemma-3-12b-it",
    "provider": "gemini",
    "input_cost_per_million": 0.00,
    "output_cost_per_million": 0.00,
    "pricing_source": "Gemini API pricing page, May 2025"
  },
  {
    "name": "Gemma 3 27B",
    "model_id": "gemini/gemma-3-27b-it",
    "provider": "gemini",
    "input_cost_per_million": 0.10,
    "output_cost_per_million": 0.30,
    "pricing_source": "Nebius Pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Qwen 3.2 235B",
    "model_id": "huggingface/Qwen/Qwen3-235B-A22B",
    "provider": "huggingface",
    "input_cost_per_million": 0.025,
    "output_cost_per_million": 0.05,
    "pricing_source": "Hugging Face Inference API pricing, June 2024 - updated 5/19/2025"
  },
  {
    "name": "Microsoft Phi 4",
    "model_id": "huggingface/microsoft/phi-4",
    "provider": "huggingface",
    "input_cost_per_million": 0.025,
    "output_cost_per_million": 0.05,
    "pricing_source": "Hugging Face Inference API pricing, June 2024 - updated 5/19/2025"
  },
  {
    "name": "Command A",
    "model_id": "cohere/command-a-03-2025",
    "provider": "cohere",
    "input_cost_per_million": 2.50,
    "output_cost_per_million": 10.00,
    "pricing_source": "Cohere API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Command R 7B",
    "model_id": "cohere/command-r7b-12-2024",
    "provider": "cohere",
    "input_cost_per_million": 0.0375,
    "output_cost_per_million": 0.15,
    "pricing_source": "Cohere API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Command R",
    "model_id": "cohere/command-r",
    "provider": "cohere",
    "input_cost_per_million": 0.15,
    "output_cost_per_million": 0.60,
    "pricing_source": "Cohere API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Command R+",
    "model_id": "cohere/command-r-plus",
    "provider": "cohere",
    "input_cost_per_million": 3.00,
    "output_cost_per_million": 15.00,
    "pricing_source": "Cohere API pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Mixtral 8x7B Instruct",
    "model_id": "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "provider": "huggingface",
    "input_cost_per_million": 0.025,
    "output_cost_per_million": 0.05,
    "pricing_source": "Hugging Face Inference API pricing, June 2024 - updated 5/19/2025"
  },
  {
    "name": "Microsoft Phi 3.5 Mini Instruct",
    "model_id": "huggingface/microsoft/Phi-3.5-mini-instruct",
    "provider": "huggingface",
    "input_cost_per_million": 0.025,
    "output_cost_per_million": 0.05,
    "pricing_source": "Hugging Face Inference API pricing, June 2024 - updated 5/19/2025"
  },
  {
    "name": "Qwen 3 32B",
    "model_id": "huggingface/Qwen/Qwen3-32B",
    "provider": "huggingface",
    "input_cost_per_million": 0.025,
    "output_cost_per_million": 0.05,
    "pricing_source": "Hugging Face Inference API pricing, June 2024 - updated 5/19/2025"
  },
  {
    "name": "Mistral Nemo Instruct 2407",
    "model_id": "huggingface/mistralai/Mistral-Nemo-Instruct-2407",
    "provider": "huggingface",
    "input_cost_per_million": 0.25,
    "output_cost_per_million": 0.25,
    "pricing_source": "Galaxy.ai model pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Deepseek V3",
    "model_id": "together_ai/deepseek-ai/DeepSeek-V3",
    "provider": "together_ai",
    "input_cost_per_million": 1.25,
    "output_cost_per_million": 7.00,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  },
  {
    "name": "Mistral 7B Instruct",
    "model_id": "together_ai/mistralai/Mistral-7B-Instruct-v0.3",
    "provider": "together_ai",
    "input_cost_per_million": 0.25,
    "output_cost_per_million": 0.25,
    "pricing_source": "Groq pricing page, May 2025 - updated 5/19/2025"
  }
]


# Additional model configurations can be added here if needed 

MODEL_CAPS = {
    "max_matches_per_model": 20,      # Maximum matches a model can play in total
}

# For backward compatibility with code that expects cost_per_million
for model in MODELS:
    if "cost_per_million" not in model:
        model["cost_per_million"] = model["output_cost_per_million"]