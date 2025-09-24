# Mathematical Reasoning Language Model Elo Rating

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green.svg)](https://www.mongodb.com/)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-orange.svg)](https://github.com/BerriAI/litellm)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)

*A peer-federated evaluation framework addressing benchmark overfitting and cost-efficiency trade-offs in large language model assessment*

[Results](#-results) • [Quick Start](#-quick-start) • [Installation](#-installation)

</div>

---

```
DETAILED LEADERBOARD - Mathematical Reasoning Tournament Results
┌───┬──────────────────────────────────┬────────┬────────┬──────────┬──────────┬────────────┬────────┬───────────┬────────┐
│ # │Model                             │Raw ELO │Cost ELO│ Raw Avg  │ Cost Avg │ W-L-D      │ Tokens │ Cost $    │Matches │
├───┼──────────────────────────────────┼────────┼────────┼──────────┼──────────┼────────────┼────────┼───────────┼────────┤
│ 1 │GPT-o4-mini                       │ 1571.2 │ 1570.3 │  0.9459  │  0.9382  │ 51-4-0     │ 645572 │ $1.52775  │   11   │
│ 2 │GPT-o3-mini                       │ 1538.0 │ 1533.2 │  0.8576  │  0.8071  │ 30-5-0     │ 350973 │ $7.08507  │   7    │
│ 3 │Qwen 3 32B                        │ 1537.1 │ 1537.8 │  0.7889  │  0.7938  │ 34-8-3     │ 537244 │ $0.01924  │   9    │
│ 4 │GPT-4.1 mini                      │ 1531.3 │ 1532.6 │  0.7861  │  0.7988  │ 25-5-5     │ 581935 │ $0.44733  │   7    │
│ 5 │Grok 3 Mini Fast                  │ 1529.0 │ 1530.1 │  0.7576  │  0.7654  │ 28-8-3     │ 296422 │ $0.23720  │   8    │
│ 6 │GPT-o3                            │ 1524.7 │ 1519.5 │  0.9458  │  0.9124  │ 50-1-4     │ 592718 │ $14.87402 │   11   │
│ 7 │Grok 3 Mini                       │ 1521.8 │ 1521.9 │  0.6763  │  0.6760  │ 25-12-2    │ 168843 │ $0.03434  │   8    │
│ 8 │Grok 3                            │ 1520.0 │ 1518.5 │  0.6281  │  0.6196  │ 34-20-1    │ 408658 │ $2.86888  │   11   │
│ 9 │Claude 3.5 Haiku                  │ 1519.6 │ 1519.2 │  0.6548  │  0.6513  │ 29-15-1    │ 146254 │ $0.18823  │   9    │
│10 │GPT-4.1                           │ 1519.0 │ 1521.3 │  0.6713  │  0.6931  │ 21-9-5     │ 504876 │ $1.86800  │   7    │
│11 │Meta LLama 4 Maverick Instruct 17B│ 1514.7 │ 1515.7 │  0.6680  │  0.6772  │ 30-15-0    │ 25773  │ $0.00277  │   9    │
│12 │Claude 3 Opus                     │ 1514.5 │ 1508.5 │  0.6261  │  0.5739  │ 24-14-1    │ 296314 │ $6.80181  │   8    │
│13 │DeepSeek R1 Distill Llama 70B    │ 1514.0 │ 1514.2 │  0.6011  │  0.6044  │ 26-17-2    │ 74835  │ $0.06732  │   9    │
│14 │Grok 3 Fast                       │ 1507.8 │ 1503.2 │  0.5425  │  0.5128  │ 26-21-8    │ 118245 │ $1.38207  │   11   │
│15 │Gemini 2.0 Flash                  │ 1507.2 │ 1507.8 │  0.5587  │  0.5651  │ 23-18-3    │ 91137  │ $0.01453  │   9    │
│16 │Mistral Saba 24B                  │ 1500.9 │ 1500.8 │  0.5174  │  0.5156  │ 9-9-1      │ 11253  │ $0.00889  │   4    │
│17 │Meta LLama 4 Scout Instruct 17B   │ 1498.5 │ 1498.7 │  0.4982  │  0.5008  │ 19-19-2    │ 66809  │ $0.00590  │   8    │
│18 │GPT-4o                            │ 1498.2 │ 1493.3 │  0.5076  │  0.4782  │ 28-27-5    │ 287981 │ $2.13102  │   12   │
│19 │Qwen 3.2 235B                     │ 1496.6 │ 1497.3 │  0.5453  │  0.5511  │ 24-20-1    │ 75909  │ $0.00293  │   9    │
│20 │Microsoft Phi 4                   │ 1495.1 │ 1496.6 │  0.4704  │  0.4811  │ 21-24-5    │ 86930  │ $0.00266  │   10   │
│21 │Gemini 2.0 Flash Lite             │ 1493.1 │ 1493.5 │  0.4495  │  0.4509  │ 19-23-2    │ 10547  │ $0.00149  │   9    │
│22 │Grok 2                            │ 1489.2 │ 1489.1 │  0.4219  │  0.4221  │ 18-25-2    │ 64056  │ $0.36185  │   9    │
│23 │Gemma 3 27B                       │ 1488.1 │ 1491.1 │  0.3977  │  0.4269  │ 15-23-1    │ 36755  │ $0.00583  │   8    │
│24 │GPT-4.1 nano                      │ 1485.9 │ 1492.2 │  0.4510  │  0.4987  │ 23-26-1    │ 48845  │ $0.00767  │   10   │
│25 │Gemma 3 12B                       │ 1485.0 │ 1486.6 │  0.3956  │  0.4083  │ 18-27-0    │ 13337  │ $0.00000  │   9    │
│26 │Gemma 3 4B                        │ 1475.8 │ 1476.1 │  0.2574  │  0.2604  │ 9-26-0     │ 11411  │ $0.00000  │   7    │
│27 │Claude 3.5 Sonnet                 │ 1475.6 │ 1475.0 │  0.2968  │  0.2923  │ 9-26-3     │ 27471  │ $0.14046  │   8    │
│28 │Gemma 2 9B                        │ 1468.7 │ 1469.2 │  0.2860  │  0.2906  │ 10-24-0    │ 42799  │ $0.00856  │   7    │
│29 │Gemini 1.5 Flash 8B               │ 1465.3 │ 1468.8 │  0.2326  │  0.2588  │ 10-34-1    │ 14102  │ $0.00087  │   9    │
│30 │Claude 3 Haiku                    │ 1463.0 │ 1466.0 │  0.1767  │  0.2048  │ 5-31-4     │ 17058  │ $0.00862  │   8    │
│31 │GPT-3.5 Turbo                     │ 1462.4 │ 1462.4 │  0.1566  │  0.1557  │ 5-28-1     │ 28949  │ $0.01761  │   7    │
│32 │LLaMA 3.1 8B Instant              │ 1461.8 │ 1462.0 │  0.2646  │  0.2651  │ 13-39-3    │ 34788  │ $0.00202  │   11   │
│33 │Llama 3.3 70B                     │ 1448.1 │ 1448.5 │  0.1450  │  0.1468  │ 6-41-2     │ 22915  │ $0.01436  │   10   │
│34 │Allamanda 2 7B                    │ 1446.9 │ 1447.0 │  0.0000  │  0.0000  │ 0-35-0     │ 57477  │ $0.00832  │   7    │
│35 │Gemma 3 1B                        │ 1442.0 │ 1442.0 │  0.0238  │  0.0245  │ 1-39-0     │ 12268  │ $0.00000  │   8    │
└───┴──────────────────────────────────┴────────┴────────┴──────────┴──────────┴────────────┴────────┴───────────┴────────┘
```

Detailed Logs have been uploaded to google drive

[Detailed Logs](https://drive.google.com/drive/folders/19368lLGwuYc7eqgG9wAnG45B1mnmkLnB?usp=sharing)

*The above table represents the complete mathematical reasoning tournament results, showing the actual performance rankings from our comprehensive evaluation.*

## Quick Start

### Prerequisites
- Python 3.8+
- MongoDB database
- API keys for LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/elo-benchmark-mathematical-reasoning.git
cd elo-benchmark

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and MongoDB URI
```

### Run Your First Tournament

```bash
# Start a tournament with default settings
python main.py

# Run with custom parameters
python main.py --batch-size 10 --stats

# Check model health
python check_models.py
```

## Installation

### Environment Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/elo-benchmark-mathematical-reasoning.git
cd elo-benchmark
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
Create a `.env` file:
```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_KEY=your_huggingface_key
GROQ_API_KEY=your_groq_key
XAI_API_KEY=your_xai_key

# Database Configuration
MONGODB_URI=your_mongodb_connection_string
```

### Model Configuration

Add new models in `model_definitions.py`:

```python
{
    "name": "GPT-4 Turbo",
    "model_id": "gpt-4-turbo-preview",
    "provider": "openai",
    "input_cost_per_million": 10.0,
    "output_cost_per_million": 30.0,
    "pricing_source": "OpenAI API Pricing"
}
```

## Usage

### Command Line Interface

```bash
# Basic tournament run
python main.py

# Advanced options
python main.py \
    --max-matches 100 \
    --batch-size 10 \
    --stats \
    --log-level INFO
```

### Programmatic Usage

```python
from models import initialize_models
from tournament import run_tournament_matches
from model_definitions import MODELS

# Initialize models
models = initialize_models(MODELS)

# Run tournament
matches = run_tournament_matches(models, max_matches=50)

# Analyze results
for match in matches:
    print(f"Match: {match.participants}")
    print(f"Scores: {match.judgment['raw_score']}")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_matches` | 50 | Maximum matches per model |
| `batch_size` | 10 | Matches per batch |
| `k_factor` | 32 | Elo update rate |
| `cost_temperature` | 0.05 | Cost sensitivity |
| `judge_temperature` | 300 | Judge weight temperature |

## Project Structure

```
elo-benchmark/
├── main.py                    # Main entry point
├── tournament.py              # Tournament management and pairing
├── models.py                  # LLM model classes and Elo tracking
├── matches.py                 # Match logic and prompt templates
├── database.py                # MongoDB operations and data persistence
├── config.py                  # Configuration and environment variables
├── model_definitions.py       # Model specifications and pricing
├── leaderboard.py             # Results display and ranking
├── check_models.py            # Model health checks and validation
├── match_results_table.py     # Results analysis and visualization
├── logger_config.py           # Logging configuration
├── logs/                      # Tournament logs and match history
├── tests/                     # Unit tests and integration tests
└── docs/                      # Additional documentation
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/elo-benchmark-mathematical-reasoning.git
cd elo-benchmark
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Check code style
black . && flake8 . && mypy .
```

### Research Contributions

We particularly welcome:
- Novel evaluation methodologies
- Mathematical improvements to the rating system
- New domain adaptations
- Efficiency optimizations
- Empirical studies and analysis

## License

This project is licensed under the ([![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
).

## Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM access
- Inspired by chess Elo rating systems and [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- Mathematical framework based on [Bradley-Terry models](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- Mathematical evaluation methodology inspired by educational assessment standards
- Special thanks to the open-source AI research community


<div align="center">

</div>
