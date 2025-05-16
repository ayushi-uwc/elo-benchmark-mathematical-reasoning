# UNER: United Nasscom Elo Rating

## Overview

UNER (United Nasscom Elo Rating) is an advanced, peer-federated evaluation framework designed to address two critical challenges in assessing large-scale generative AI models: **benchmark overfitting** and **cost-efficiency trade-offs**. Traditional static benchmarks encourage models to game narrow test sets over time, while performance-focused metrics ignore substantial computational, latency, and energy costs at scale. UNER overcomes these limitations by dynamically generating fresh evaluation challenges through a peer-driven process and by maintaining **dual-track Elo ratings**—one for raw capability and another cost-adjusted—to capture both quality and efficiency.

Key innovations in UNER include:  

- **Dynamic Challenge Generation**: Each cycle begins with top-performing models constructing novel prompts via stratified topic sampling, thwarting overfitting on fixed corpora.
- **Peer-Federated Judging**: Models themselves act as judges. Votes are weighted by judges’ current Elo ratings to prioritize reliable evaluators.
- **Dual-Track Elo Ratings**:
  - **Raw Performance** ($R^{\mathrm{raw}}$): Quality-focused rating.
  - **Cost-Adjusted Performance** ($R^{\mathrm{cost}}$): Balances quality against resource consumption.

  Mathematically, for a match between models $A$ and $B$:
  
  \[
    egin{aligned}
      E^{\mathrm{raw}}_A &= \frac{1}{1 + 10^{\frac{R^{\mathrm{raw}}_B - R^{\mathrm{raw}}_A}{400}}}, \
      E^{\mathrm{cost}}_A &= \frac{1}{1 + 10^{\frac{R^{\mathrm{cost}}_B - R^{\mathrm{cost}}_A}{400}}}.
    \end{aligned}
  \]

- **Swiss-Style Tournament Pairing**: Models are matched within a cost-adjusted Elo window of width $\Delta$ (e.g., 50 points) to ensure fair competition.
- **Immutable, Transparent Logging**: All match artifacts—prompts, responses, votes, costs—are recorded in a publicly auditable ledger for reproducibility.

This domain-agnostic framework adapts easily to new tasks (vision, speech, multimodal) by updating prompt templates and cost metrics. Its modular Python code integrates seamlessly into existing pipelines.

## Methodology

UNER’s evaluation pipeline consists of eight stages, each combining statistical rigor, computational feasibility, and robustness against overfitting.

### 1. Model Registration

- **Objective**: Initialize each model’s metadata and Elo scores.  
- **Procedure**:  
  1. Load model definitions (architecture, cost-per-token, API endpoint) from `configs/models.yaml`.  
  2. Set initial scores:  
     \[
       R^{\mathrm{raw}}_m = R^{\mathrm{cost}}_m = R_0 
       \quad\text{for all models } m,
     \]
     where $R_0$ (e.g., 1500) is the baseline Elo.  
  3. Store metadata for runtime cost tracking.

```python
for m in models:
    R_raw[m] = R0
    R_cost[m] = R0
    metadata[m] = load_config(m)
```

### 2. Fair Matching (Swiss Pairing + Stratified Sampling)

- **Objective**: Pair models to maximize information gain while ensuring comparable cost profiles.  
- **Algorithm**:  
  1. Sort models by $R^{\mathrm{cost}}$.  
  2. Partition into strata of width $\Delta = 50$ Elo points.  
  3. Within each stratum, apply Swiss-style pairing to minimize Elo differences.

### 3. Challenge Creation & Validation

- **Objective**: Generate a novel, solvable prompt for each pairing.  
- **Process**:  
  1. Higher-seeded model drafts a prompt.  
  2. A committee of top-$k$ models evaluates prompt clarity via voting.  
  3. Repeat until at least 80% approval is reached.

### 4. Solution Generation

- **Objective**: Obtain responses and accurately record resource usage.  
- **Mechanics**:  
  - Use standardized decoding (e.g., temperature, max tokens).  
  - Log compute cost $C_m$ (e.g., GPU-seconds) and latency for each model $m$.  
  - Normalize costs to a unified unit.

### 5. Peer Evaluation

- **Objective**: Determine match outcomes through weighted voting.  
- **Method**:  
  1. Select top-$j$ models by $R^{\mathrm{raw}}$ (excluding contestants) as judges.  
  2. Each judge casts a binary vote comparing outputs of $A$ vs. $B$.  
  3. Assign weight $w_i \propto R^{\mathrm{raw}}_i$ to judge $i$.  
  4. Compute contestant $A$’s raw score:  
     \[
       S_A = 
       egin{cases}
         1 & \text{if } A\text{ wins},\\
         0.5 & \text{if tie},\\
         0 & \text{otherwise}
       \end{cases}
     \]

### 6. Dual Scoring & Elo Updates

- **Adjusted Score**:  
  \[
    S^{\mathrm{adj}}_A = S_A - \tau_c \cdot \frac{C_A}{C_A + C_B},
  \]
  where $\tau_c$ controls cost sensitivity.

- **Expected Scores**:  
  \[
    \begin{aligned}
      E^{\mathrm{raw}}_A &= \frac{1}{1 + 10^{(R^{\mathrm{raw}}_B - R^{\mathrm{raw}}_A)/400}}, \\
      E^{\mathrm{cost}}_A &= \frac{1}{1 + 10^{(R^{\mathrm{cost}}_B - R^{\mathrm{cost}}_A)/400}}.
    \end{aligned}
  \]

- **Elo Updates**:  
  \[
    \begin{aligned}
      R^{\mathrm{raw}}_A &\leftarrow R^{\mathrm{raw}}_A + K \\bigl(S_A - E^{\mathrm{raw}}_A\bigr), \\
      R^{\mathrm{cost}}_A &\leftarrow R^{\mathrm{cost}}_A + K \\bigl(S^{\mathrm{adj}}_A - E^{\mathrm{cost}}_A\bigr).
    \end{aligned}
  \]

### 7. Rating Propagation

- **Objective**: Smooth abrupt rating changes by diffusing updates across similar models.  
- **Technique**: Apply a Gaussian kernel $G(d)$ over pairwise cost-adjusted Elo distances $d_{mn}$:  
  \[
    \Delta R_m \leftarrow \sum_{n} G\bigl(d_{mn}\bigr)\,\Delta R_n,
  \]
  ensuring ratings evolve cohesively.

### 8. Logging & Dissemination

- **Objective**: Provide full transparency and enable downstream analysis.  
- **Implementation**:  
  - Append all artifacts (prompts, responses, votes, costs, rating changes) to an immutable JSON log.  
  - Generate periodic leaderboards, Elo trajectories, and cost-vs-performance frontiers.

By continuously iterating this pipeline, UNER fosters a self-improving ecosystem that drives both generative quality and operational efficiency.


## Key Features

- **Match-Based Evaluation**: Models play matches against opponents of similar ELO ratings
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

### Running Matches

```bash
python main.py
```

### Options

```
--max-matches    Maximum total number of matches to run
--batch-size     Number of matches per batch for status updates
--stats          Show detailed model statistics
--stats-only     Only show statistics, don't run matches
--log-level      Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
--install-deps   Check and install dependencies
```

### Example

```bash
python main.py --batch-size 10 --stats
```

## How It Works

### Match Structure

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
- `tournament.py` - Match management logic
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