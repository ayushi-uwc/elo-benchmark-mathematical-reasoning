# ELO Benchmark

A tournament-based framework for benchmarking Large Language Models (LLMs) using an ELO rating system.

## Overview

This project implements a robust system for comparing and benchmarking LLMs through tournament-style matches. Models compete against each other in clinical reasoning tasks, with peer models acting as judges. The system uses an ELO rating system with both raw and cost-adjusted scores to evaluate model performance.

## Key Features

- **Tournament Structure**: Models play matches against opponents of similar ELO ratings
- **Federated Judgment**: Peer models evaluate responses with votes weighted by judge ELO
- **Dual ELO Ratings**: Track both raw performance and cost-adjusted efficiency
- **Standardized Judging**: Judges must provide verdicts in a specific format
- **MongoDB Integration**: Store and track model performance over time

## Requirements

- Python 3.8+
- MongoDB database
- API keys for LLM providers:
  - OpenAI
  - Anthropic
  - Google (Gemini)
  - Hugging Face
  - Groq
  - Other providers as configured

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/elo-benchmark.git
cd elo-benchmark
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys and MongoDB connection string
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_KEY=your_huggingface_key
GROQ_API_KEY=your_groq_key
XAI_API_KEY=your_xai_key_here
MONGODB_URI=your_mongodb_connection_string
```

## Usage

### Running a Tournament

```bash
python main.py
```

### Options

```
--max-matches    Maximum total number of matches to run
--batch-size     Number of matches per batch for status updates
--stats          Show detailed model statistics
--stats-only     Only show statistics, don't run tournament
--log-level      Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
--install-deps   Check and install dependencies
```

### Example

```bash
python main.py --batch-size 10 --stats
```

## How It Works

### Tournament Structure

1. **Model Pairing**: Models are paired with opponents of similar ELO ratings
2. **Case Generation**: A top-performing model generates a clinical case
3. **Question Generation**: Another top model generates a question about the case
4. **Response Collection**: Both competing models answer the question
5. **Judgment**: Multiple judge models evaluate the responses
6. **ELO Updates**: Models' ratings are updated based on the weighted votes

### ELO Rating System

- **Raw ELO**: Based purely on judge votes
- **Cost-Adjusted ELO**: Incorporates response costs into the score calculation
- **K-Factor**: 16 (configurable)
- **Initial Rating**: 1500

### Federated Judgment Process

1. Judge votes are weighted using a softmax function based on their raw ELO
2. The weight of judge k is: w_k = e^(R_k^raw/τ) / Σ_j=1^J e^(R_j^raw/τ)
3. Raw scores are derived from weighted votes
4. Cost-adjusted scores incorporate efficiency factors

## Project Structure

- `main.py` - Main entry point
- `tournament.py` - Tournament management logic
- `models.py` - LLM model implementation
- `matches.py` - Match logic and prompt templates
- `database.py` - MongoDB interaction
- `config.py` - Configuration and environment variables
- `model_definitions.py` - Model specifications

## Adding New Models

Add new models to `model_definitions.py`:

```python
{
  "name": "Model Name",
  "model_id": "provider/model-id",
  "provider": "provider",
  "input_cost_per_million": 0.50,
  "output_cost_per_million": 1.50,
  "pricing_source": "Source of pricing info"
}
```

## License

[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](LICENSE)

This work is licensed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, which allows for sharing and adaptation with attribution and sharing under the same license terms.

## Acknowledgements

This project implements the methodology described in [your methodology paper/reference]. 