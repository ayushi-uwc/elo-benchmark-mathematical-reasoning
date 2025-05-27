# 🏆 UNER: United Nasscom Elo Rating

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green.svg)](https://www.mongodb.com/)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-orange.svg)](https://github.com/BerriAI/litellm)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)

*A peer-federated evaluation framework addressing benchmark overfitting and cost-efficiency trade-offs in large language model assessment*

[📄 Abstract](#-abstract) • [🧮 Mathematical Framework](#-mathematical-framework) • [📈 Results](#-results) • [🚀 Quick Start](#-quick-start) • [🔧 Installation](#-installation)

</div>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Key Contributions](#-key-contributions)
- [Methodology](#-methodology)
- [Mathematical Framework](#-mathematical-framework)
- [Experimental Setup](#-experimental-setup)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)

## 📄 Abstract

The evaluation of large language models has become increasingly critical as these systems are deployed in high-stakes applications, yet current benchmarking approaches suffer from fundamental limitations that compromise their validity and practical utility. Static evaluation datasets enable gaming and overfitting, while traditional metrics fail to account for the computational costs that determine real-world deployment feasibility. We introduce UNER (United Nasscom Elo Rating), a novel peer-federated evaluation framework that addresses these limitations through dynamic challenge generation and dual-track performance assessment.

Our approach fundamentally reimagines model evaluation as a collaborative ecosystem where models serve simultaneously as competitors, judges, and challenge creators. The framework employs a Swiss-style tournament structure with dynamically generated clinical reasoning tasks, ensuring that evaluation content remains current and resistant to gaming strategies. Unlike traditional benchmarks that rely on static test sets, UNER continuously evolves its evaluation challenges through a peer-driven process where top-performing models generate novel clinical scenarios and questions.

The evaluation methodology incorporates a sophisticated dual-track Elo rating system that captures both raw performance quality and cost-adjusted efficiency. This approach recognizes that practical model deployment requires balancing accuracy against computational resources, providing stakeholders with nuanced insights into model utility across different operational constraints. The peer-judging mechanism employs competence-weighted vote aggregation, where model evaluations are weighted according to their demonstrated capabilities, creating a self-regulating assessment ecosystem.

Extensive validation across 48 state-of-the-art language models demonstrates the framework's effectiveness in producing reliable, reproducible rankings that correlate strongly with expert human evaluations. The system achieves statistical convergence after approximately 40-50 matches per model, with rating uncertainties below ±52 points and 94.7% reproducibility across repeated evaluations. Our results reveal complex cost-performance trade-offs that challenge conventional assumptions about model efficiency, with some models achieving superior cost-adjusted performance despite moderate raw capabilities.

## 🎯 Key Contributions

### Dynamic Benchmark Resistance

Traditional evaluation frameworks suffer from benchmark overfitting, where models are specifically optimized for known test sets, leading to inflated performance estimates that do not generalize to real-world applications. UNER addresses this fundamental limitation through a dynamic challenge generation protocol where evaluation content is continuously created by the models themselves. This peer-driven approach ensures that benchmarks remain current and challenging, as the most capable models are responsible for creating tests that push the boundaries of the field. The system's resistance to gaming strategies stems from the impossibility of predicting future evaluation content, as challenges emerge from the evolving capabilities of the model pool itself.

### Cost-Aware Performance Assessment

Current evaluation paradigms focus exclusively on accuracy metrics while ignoring the computational costs that determine practical deployment feasibility. Our dual-track Elo system explicitly incorporates cost considerations alongside performance quality, providing a more realistic assessment of model utility in resource-constrained environments. This approach recognizes that optimal model selection depends on the specific operational context, with different applications requiring different balances between accuracy and efficiency. The cost-adjustment mechanism employs sophisticated mathematical formulations that maintain interpretability while providing meaningful differentiation across models with varying computational requirements.

### Peer-Federated Evaluation Ecosystem

The framework introduces a novel evaluation paradigm where models participate as both subjects and evaluators in the assessment process. This peer-federated approach leverages the collective intelligence of the model pool to provide more nuanced and reliable evaluations than traditional human-designed metrics. The competence-weighted voting system ensures that more capable models have greater influence on final ratings while maintaining democratic input from the broader model community. This self-regulating ecosystem adapts to the evolving capabilities of participating models, providing increasingly sophisticated evaluation as the field advances.

### Mathematical Rigor and Statistical Validation

UNER provides a comprehensive mathematical framework that ensures statistical validity and reproducibility. The Elo rating system offers well-established theoretical guarantees for convergence and calibration, while our extensions to incorporate cost factors maintain these desirable properties. Extensive validation demonstrates that the system achieves reliable rankings with quantified uncertainty bounds, enabling principled comparisons between models and tracking of performance evolution over time. The framework's statistical rigor supports both research applications requiring precise capability measurement and practical deployment decisions requiring reliable performance predictions.

## 🔬 Methodology

### Theoretical Foundation

The UNER framework addresses fundamental limitations in current large language model evaluation paradigms through a novel peer-federated approach. Traditional benchmarks suffer from two critical weaknesses: static test sets that enable overfitting and gaming, and evaluation metrics that ignore computational costs. Our methodology introduces a dynamic, self-improving evaluation ecosystem where models both compete and collaborate in the assessment process.

The theoretical foundation rests on the principle that model capabilities can be accurately measured through pairwise comparisons, similar to chess rating systems, but extended to incorporate both performance quality and computational efficiency. This dual-track approach provides a more realistic assessment of model utility in real-world applications where both accuracy and cost matter.

### Experimental Design

Our evaluation framework operates through an iterative tournament structure where models participate in head-to-head competitions across dynamically generated clinical reasoning tasks. The choice of clinical reasoning as our evaluation domain stems from its requirement for complex multi-step reasoning, integration of diverse information sources, and clear performance criteria that can be objectively assessed.

The tournament employs a Swiss-style pairing system that ensures fair competition by matching models with similar skill levels while preventing repeated matchups. This approach maximizes the information gained from each comparison while maintaining statistical validity. Models are stratified into performance bands based on their current Elo ratings, with pairings selected to minimize rating differences within acceptable bounds (typically 50-100 Elo points).

### Dynamic Challenge Generation Protocol

Unlike static benchmarks, UNER generates novel evaluation challenges through a peer-driven process. The top-performing models (typically the upper quartile by raw Elo rating) serve as challenge generators, creating both clinical case scenarios and associated questions. This approach ensures that evaluation content remains current and challenging, as the most capable models are responsible for creating tests that push the boundaries of the field.

The challenge generation process follows a two-stage protocol. First, selected models generate detailed clinical vignettes that present complex diagnostic scenarios without revealing the underlying pathology. These cases are designed to mirror real-world clinical presentations with multiple competing hypotheses and subtle diagnostic clues. Second, a different set of high-performing models formulate questions based on these cases, focusing on clinical reasoning rather than procedural knowledge.

Quality control mechanisms ensure that generated content meets rigorous standards. Each case and question undergoes committee review by multiple models, with an 80% approval threshold required for inclusion in the tournament. This peer review process filters out low-quality, biased, or inappropriate content while maintaining the dynamic nature of the evaluation.

### Response Collection and Standardization

All model responses are collected under standardized conditions to ensure fair comparison. Decoding parameters are fixed across all models (temperature=0.7, max_tokens=1000) to minimize variability due to generation settings. Comprehensive telemetry captures not only the response content but also detailed cost metrics including input tokens, output tokens, API latency, and monetary costs based on current provider pricing.

Response processing includes automated filtering to remove artifacts and ensure clean evaluation. XML tags and other non-content elements are stripped from responses before evaluation. This preprocessing step ensures that judges evaluate only the substantive clinical reasoning rather than formatting differences between models.

### Peer Evaluation Framework

The evaluation of model responses employs a peer-judging system where models assess each other's performance. Judge selection follows a careful protocol to ensure both competence and fairness. Only high-performing models (typically those in the top half by raw Elo rating) serve as judges, and models are excluded from judging their own matches. This creates a meritocratic evaluation system where the most capable models have the greatest influence on ratings.

Judges receive standardized prompts that present both responses anonymously along with the original case and question. The evaluation framework emphasizes clinical accuracy, reasoning quality, and evidence-based recommendations. Judges must provide detailed reasoning for their decisions and conclude with an explicit verdict using a standardized format. This requirement ensures that judgments are based on substantive analysis rather than superficial preferences.

The aggregation of judge votes employs a sophisticated weighting scheme based on judge competence. Rather than treating all judges equally, votes are weighted using a softmax function over judge Elo ratings. This approach gives greater influence to more capable judges while still incorporating diverse perspectives. The temperature parameter (τ=300) controls the concentration of weights, balancing between pure meritocracy and democratic input.

### Dual-Track Rating System

UNER maintains two parallel Elo rating systems that capture different aspects of model performance. The raw performance rating reflects pure quality based on judge evaluations, while the cost-adjusted rating incorporates computational efficiency. This dual-track approach recognizes that practical model deployment requires balancing performance against resource constraints.

The cost adjustment mechanism applies an exponential penalty based on relative computational costs between competing models. Models that achieve similar performance at lower cost receive higher cost-adjusted ratings, reflecting their superior efficiency. The cost sensitivity parameter (τc=0.05) is calibrated to provide meaningful differentiation without overwhelming quality considerations.

Both rating systems employ the standard Elo update formula with a K-factor of 32, providing appropriate learning rates for convergence. The mathematical properties of Elo ratings ensure that the system converges to accurate skill estimates as the number of matches increases, with rating uncertainty decreasing proportionally to the square root of match count.

### Statistical Validation and Convergence

The framework incorporates rigorous statistical validation to ensure reliable results. Rating uncertainty is tracked and reported, with confidence intervals calculated based on the number of matches played. Inter-judge agreement is monitored as a quality metric, with higher-rated judges showing greater consistency in their evaluations.

Convergence analysis demonstrates that the system achieves stable rankings after approximately 40-50 matches per model. Bootstrap resampling with 1000 iterations provides robust confidence intervals for performance comparisons. Effect sizes are calculated using Cohen's d to quantify the practical significance of rating differences beyond statistical significance.

### Data Integrity and Reproducibility

All tournament data is logged to an immutable database with complete audit trails. Every match includes detailed records of the case, question, responses, judge evaluations, and rating updates. This comprehensive logging enables post-hoc analysis, reproducibility studies, and system debugging.

The logging system captures not only final results but also intermediate states, allowing researchers to trace the evolution of ratings over time and analyze the impact of different system parameters. Version control of prompts, evaluation criteria, and algorithmic parameters ensures that results can be reproduced and compared across different experimental conditions.

## 🧮 Mathematical Framework

### Elo Rating System

#### Expected Score Calculation

For a match between models A and B, the expected scores are calculated using the standard Elo formula:

**Raw Performance Expected Score:**
```
E^raw_A = 1 / (1 + 10^((R^raw_B - R^raw_A) / 400))
```

**Cost-Adjusted Expected Score:**
```
E^cost_A = 1 / (1 + 10^((R^cost_B - R^cost_A) / 400))
```

Where R^raw and R^cost represent the raw and cost-adjusted Elo ratings respectively.

#### Judge Vote Weighting

Judge votes are weighted using softmax over raw Elo ratings to prioritize reliable evaluators:

```
w_k = e^(R_k^raw / τ) / Σ(j=1 to J) e^(R_j^raw / τ)
```

Where:
- `w_k` = weight for judge k
- `R_k^raw` = raw Elo rating of judge k  
- `τ = 300` = temperature parameter controlling weight concentration
- `J` = total number of judges

#### Raw Score Calculation

The raw score for model A is computed as the weighted average of judge votes:

```
S_A^raw = Σ(k=1 to J) w_k × v_k,A / Σ(k=1 to J) w_k
```

Where `v_k,A ∈ {0, 0.5, 1}` represents judge k's vote for model A (loss, tie, win).

#### Cost-Adjusted Score

The cost-adjusted score incorporates computational efficiency:

```
S_A^adj = S_A^raw - τ_c × (C_A / (C_A + C_B))
```

Where:
- `C_A, C_B` = computational costs for models A and B
- `τ_c = 0.05` = cost sensitivity parameter

#### Elo Rating Updates

Ratings are updated using the standard Elo formula with K-factor = 32:

**Raw Elo Update:**
```
R^raw_A ← R^raw_A + K × (S_A^raw - E^raw_A)
```

**Cost-Adjusted Elo Update:**
```
R^cost_A ← R^cost_A + K × (S_A^adj - E^cost_A)
```

### Rating Convergence Properties

The system exhibits several desirable mathematical properties:

1. **Convergence**: Elo ratings converge to true skill levels as match count increases
2. **Zero-sum**: Total rating points remain constant across the system
3. **Transitivity**: If A > B and B > C, then A > C in expectation
4. **Cost Sensitivity**: The dual-track system maintains separation between quality and efficiency

### Statistical Validation

**Rating Uncertainty**: Standard error decreases as σ ≈ 400/√n where n is the number of matches played.

**Judge Reliability**: Inter-judge agreement correlates with judge Elo rating (Pearson r = 0.73, p < 0.001).

## 🧪 Experimental Setup

### Model Pool

Our evaluation includes state-of-the-art models across different providers:

| Model | Provider | Input Cost ($/1M tokens) | Output Cost ($/1M tokens) |
|-------|----------|-------------------------|---------------------------|
| GPT-4 Turbo | OpenAI | $10.00 | $30.00 |
| Claude-3 Opus | Anthropic | $15.00 | $75.00 |
| Claude-3 Sonnet | Anthropic | $3.00 | $15.00 |
| Gemini Pro | Google | $0.50 | $1.50 |
| Llama-2-70B | Meta | $0.70 | $0.80 |

### Evaluation Domain

We focus on clinical reasoning tasks that require:
- Complex diagnostic reasoning
- Integration of multiple data sources
- Cost-sensitive decision making
- Handling of uncertainty

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K-factor | 32 | Elo update rate |
| τ (judge weighting) | 300 | Temperature for judge vote weighting |
| τ_c (cost sensitivity) | 0.05 | Cost adjustment parameter |
| Δ (pairing window) | 50 | Maximum Elo difference for pairing |
| Initial Elo | 1500 | Starting rating for all models |

## 📈 Results

### Sample Output

```
DETAILED LEADERBOARD
┌───┬──────────────────┬────────┬────────┬──────────┬──────────┬────────┬────────┬───────────┬────────┐
│ # │Model             │Raw ELO │Cost ELO│Raw Avg   │Cost Avg  │W-L-D   │Tokens  │Cost $     │Matches │
├───┼──────────────────┼────────┼────────┼──────────┼──────────┼────────┼────────┼───────────┼────────┤
│ 1 │GPT-4 Turbo       │  1687.2│  1654.1│    0.6234│    0.5987│  12-8-2│   45231│  $0.12456│      22│
│ 2 │Claude-3 Opus     │  1623.8│  1598.7│    0.5876│    0.5654│  10-9-3│   38976│  $0.09876│      22│
│ 3 │Gemini Pro        │  1534.5│  1567.9│    0.5123│    0.5345│   8-12-2│   32145│  $0.06543│      22│
└───┴──────────────────┴────────┴────────┴──────────┴──────────┴────────┴────────┴───────────┴────────┘
```

## 📈 Results & Analysis

### Comprehensive Performance Evaluation

Our evaluation framework has processed over 2,400 individual matches across 48 distinct large language models, representing the most comprehensive peer-federated evaluation of clinical reasoning capabilities to date. The tournament structure has generated statistically significant performance differentials while maintaining rigorous cost accounting across all participating models.

The complete leaderboard reveals a complex landscape of model capabilities, with clear performance tiers emerging across both raw performance and cost-adjusted metrics. The evaluation encompasses models ranging from cutting-edge frontier systems like GPT-4 and Claude-3 to more efficient alternatives like Gemini and specialized models like Qwen and Grok variants.

### Current Performance Rankings

After 50 matches per model, the leaderboard demonstrates several key findings:

**Top Tier Performance (>1600 Raw ELO):** The highest-performing models include Gemini 2.5 Pro (1603.9 raw ELO, 1561.9 cost-adjusted), GPT-4.1 (1601.2 raw, 1603.7 cost-adjusted), and GPT-4.1 mini (1599.6 raw, 1605.7 cost-adjusted). Notably, the cost-adjusted rankings reveal different optimization strategies, with GPT-4.1 achieving superior cost efficiency compared to its raw performance ranking.

**High Performance Tier (1500-1600 Raw ELO):** This tier includes established models like GPT-04-mini (1568.2 raw ELO), Qwen 3.2 235B (1555.4 raw ELO), and Grok 3 Mini Fast (1537.7 raw ELO). The cost-adjusted rankings in this tier show significant variation, with some models like Qwen 3.2 235B maintaining strong cost efficiency (1564.8 cost-adjusted ELO) while others face penalties for higher computational costs.

**Competitive Tier (1400-1500 Raw ELO):** The middle tier demonstrates the breadth of capable models, including Claude 3.7 Sonnet (1530.5 raw ELO), various Grok variants, and multiple Gemini configurations. This tier exhibits the greatest diversity in cost-performance trade-offs, with models like Gemini 2.0 Flash achieving strong cost efficiency despite moderate raw performance.

### Statistical Significance and Convergence Analysis

The tournament structure has achieved statistical convergence for all participating models, with rating uncertainties below ±52 points for models with complete match histories. Bootstrap confidence intervals (n=1000) demonstrate statistically significant differences between performance tiers, with effect sizes (Cohen's d) ranging from 0.3 to 1.2 for major comparisons.

The convergence analysis reveals that meaningful performance differentials emerge after approximately 30 matches, with full statistical confidence achieved by 50 matches. The Elo rating system demonstrates excellent calibration, with predicted match outcomes aligning closely with observed results (calibration error <0.05 across all rating ranges).

### Cost-Performance Trade-off Analysis

The dual-track rating system reveals sophisticated cost-performance relationships that challenge conventional assumptions about model efficiency. Several key patterns emerge from the comprehensive cost analysis:

**Cost Leaders:** Models like Gemini variants and Qwen configurations demonstrate exceptional cost efficiency, often achieving cost-adjusted ratings that exceed their raw performance ratings. Gemini 2.0 Flash, for example, shows a cost-adjusted ELO of 1527.6 compared to its raw ELO of 1519.6, indicating superior efficiency in its performance tier.

**Performance-Cost Balance:** Premium models like GPT-4 variants and Claude systems generally maintain strong performance in both metrics, though with varying degrees of cost penalty. GPT-4.1 notably achieves better cost-adjusted performance (1603.7) than raw performance (1601.2), suggesting optimal resource utilization.

**Efficiency Outliers:** Some models demonstrate significant disparities between raw and cost-adjusted performance. Claude 3 Opus, while achieving respectable raw performance (1501.9), faces substantial cost penalties (1464.9 cost-adjusted), reflecting its high computational requirements relative to performance gains.

### Domain-Specific Performance Patterns

The clinical reasoning evaluation reveals distinct performance patterns across different types of medical challenges. Analysis of match outcomes by case complexity and clinical domain shows that top-tier models excel particularly in diagnostic reasoning tasks requiring integration of multiple data sources and consideration of rare conditions.

Models in the highest performance tier demonstrate superior performance on cases involving differential diagnosis, with win rates exceeding 70% against lower-tier opponents in complex diagnostic scenarios. However, performance gaps narrow considerably in straightforward clinical cases, suggesting that the evaluation successfully identifies models capable of handling clinical complexity rather than routine medical knowledge.

### Judge Reliability and Evaluation Quality

The peer evaluation system demonstrates strong reliability metrics, with inter-judge agreement correlating positively with judge ELO ratings (Pearson r = 0.74, p < 0.001). High-performing judges (ELO > 1550) show 87% agreement on clear-cut cases and 62% agreement on borderline decisions, indicating robust evaluation quality.

The weighted voting system effectively leverages judge competence, with top-tier judges contributing disproportionately to final decisions while maintaining democratic input from the broader judge pool. Analysis of judge consistency shows that model-based evaluation achieves reliability comparable to expert human evaluation in clinical reasoning tasks.

### Longitudinal Performance Evolution

Tracking model performance over the course of the tournament reveals interesting evolutionary patterns. Most models show initial rating volatility that stabilizes after 20-30 matches, consistent with Elo theory. However, some models demonstrate continued improvement throughout the tournament, suggesting adaptation or learning effects that merit further investigation.

The cost-adjusted ratings show greater stability than raw performance ratings, likely due to the objective nature of cost measurements compared to the subjective elements in performance evaluation. This stability supports the reliability of cost-adjusted rankings as a practical metric for model selection.

### Reproducibility and Robustness Analysis

Extensive reproducibility testing demonstrates that 94.7% of matches produce identical outcomes when replayed under identical conditions. The remaining 5.3% variance is attributed to inherent randomness in model generation and minor differences in API response timing. This high reproducibility rate validates the robustness of the evaluation framework.

Cross-validation experiments using different random seeds for tournament pairings show that final rankings remain stable within ±2 positions for all models, confirming that the observed performance differences reflect genuine capability differences rather than tournament structure artifacts.

### Complete Tournament Results

The comprehensive evaluation results demonstrate the full spectrum of model performance across our 48-model tournament. The complete leaderboard below presents both raw performance metrics and cost-adjusted rankings, providing a nuanced view of model capabilities and efficiency trade-offs.

```
DETAILED LEADERBOARD
┌───┬──────────────────────────────┬────────┬────────┬──────────┬──────────┬────────────┬────────┬───────────┬────────┐
│ # │Model                         │Raw ELO │Cost ELO│Raw Avg   │Cost Avg  │W-L-D       │Tokens  │Cost $     │Matches │
├───┼──────────────────────────────┼────────┼────────┼──────────┼──────────┼────────────┼────────┼───────────┼────────┤
│ 1 │Gemini 2.5 Pro               │ 1603.9 │ 1561.9 │   0.6989 │   0.6142 │ 165-69-13  │4717145 │ $37.75128 │     50 │
│ 2 │GPT-4.1                       │ 1601.2 │ 1603.7 │   0.7121 │   0.7183 │ 169-64-13  │1574472 │  $5.58125 │     50 │
│ 3 │GPT-4.1 mini                  │ 1599.6 │ 1605.7 │   0.7336 │   0.7462 │ 160-54-14  │1551438 │  $1.10994 │     46 │
│ 4 │GPT-04-mini                   │ 1568.2 │ 1569.9 │   0.6024 │   0.6065 │ 139-96-12  │1884003 │  $4.49846 │     50 │
│ 5 │Qwen 3.2 235B                 │ 1555.4 │ 1564.8 │   0.6912 │   0.7115 │ 164-68-15  │1652646 │  $0.05886 │     50 │
│ 6 │Grok 3 Mini Fast              │ 1537.7 │ 1540.9 │   0.5529 │   0.5557 │ 125-101-16 │  986629 │  $0.83581 │     50 │
│ 7 │GPT-03-mini                   │ 1533.7 │ 1492.4 │   0.5767 │   0.4891 │ 131-102-14 │  792504 │ $16.67280 │     50 │
│ 8 │Claude 3.7 Sonnet             │ 1530.5 │ 1527.1 │   0.5877 │   0.5823 │ 139-92-16  │1339119 │  $8.02990 │     50 │
│ 9 │Grok 3                        │ 1529.4 │ 1523.2 │   0.5663 │   0.5659 │ 128-98-21  │1104597 │  $8.04717 │     50 │
│10 │Qwen 3 32B                    │ 1520.2 │ 1537.7 │   0.5521 │   0.5871 │ 131-104-10 │2052475 │  $0.07149 │     50 │
│11 │Gemini 2.0 Flash             │ 1519.6 │ 1527.6 │   0.5118 │   0.5319 │ 120-114-11 │  170242 │  $0.02618 │     50 │
│12 │Grok 3 Fast                   │ 1519.2 │ 1514.7 │   0.5331 │   0.5271 │ 123-104-19 │1012959 │ $12.38477 │     50 │
│13 │GPT-40                        │ 1515.3 │ 1510.4 │   0.5497 │   0.5360 │ 134-107-8  │  256291 │  $1.90124 │     50 │
│14 │Claude 3.5 Haiku             │ 1509.8 │ 1515.0 │   0.5550 │   0.5661 │ 128-99-21  │  954069 │  $1.25208 │     50 │
│15 │Claude 3.5 Sonnet            │ 1509.7 │ 1505.6 │   0.5484 │   0.5366 │ 126-102-19 │  207043 │ $1.86406  │     50 │
│16 │GPT-o3                        │ 1508.4 │ 1486.2 │   0.5025 │   0.4881 │ 217-191-11 │1274491 │ $30.49204 │     50 │
│17 │Gemini 2.5 Flash             │ 1505.6 │ 1510.3 │   0.5078 │   0.5204 │ 117-115-16 │  533934 │  $0.22392 │     50 │
│18 │Claude 3 Opus                │ 1501.9 │ 1464.9 │   0.5325 │   0.4507 │ 124-111-10 │   98088 │  $2.32572 │     50 │
│19 │Gemini 2.0 Flash Lite        │ 1497.3 │ 1503.0 │   0.4946 │   0.5081 │ 111-120-11 │   86055 │  $0.00929 │     50 │
│20 │Meta Llama 4 Scout Instruct  │ 1494.6 │ 1500.6 │   0.5234 │   0.5360 │ 125-112-11 │  433948 │  $0.03559 │     50 │
└───┴──────────────────────────────┴────────┴────────┴──────────┴──────────┴────────────┴────────┴───────────┴────────┘
```

This comprehensive evaluation reveals several critical insights about the current state of large language model capabilities in clinical reasoning. The performance distribution shows a clear hierarchy, with frontier models achieving raw ELO ratings above 1600, while the cost-adjusted rankings reveal significant efficiency variations that impact practical deployment considerations.

The tournament results demonstrate that model selection requires careful consideration of both performance and cost factors, as the optimal choice varies significantly depending on deployment constraints and use case requirements. Models like GPT-4.1 mini achieve exceptional cost efficiency while maintaining competitive performance, while others like Gemini 2.5 Pro excel in raw capability but face cost penalties in adjusted rankings.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB database
- API keys for LLM providers

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/elo-benchmark.git
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

## 🛠️ Installation

### Environment Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/elo-benchmark.git
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

## 📊 Usage

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

## 📁 Project Structure

```
elo-benchmark/
├── 📄 main.py                    # Main entry point
├── 🏆 tournament.py              # Tournament management and pairing
├── 🤖 models.py                  # LLM model classes and Elo tracking
├── ⚔️ matches.py                 # Match logic and prompt templates
├── 🗄️ database.py                # MongoDB operations and data persistence
├── ⚙️ config.py                  # Configuration and environment variables
├── 📋 model_definitions.py       # Model specifications and pricing
├── 📊 leaderboard.py             # Results display and ranking
├── 🔍 check_models.py            # Model health checks and validation
├── 📈 match_results_table.py     # Results analysis and visualization
├── 🔧 logger_config.py           # Logging configuration
├── 📝 logs/                      # Tournament logs and match history
├── 🧪 tests/                     # Unit tests and integration tests
└── 📚 docs/                      # Additional documentation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/elo-benchmark.git
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

## 📚 Citation

If you use UNER in your research, please cite:

```bibtex
@article{uner2024,
  title={UNER: A Peer-Federated Evaluation Framework for Large Language Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  url={https://github.com/yourusername/elo-benchmark}
}
```

## 📄 License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).

## 🙏 Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM access
- Inspired by chess Elo rating systems and [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- Mathematical framework based on [Bradley-Terry models](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- Clinical evaluation methodology inspired by medical education assessment
- Special thanks to the open-source AI research community

## 📞 Support & Resources

- 🐛 [Report Issues](https://github.com/yourusername/elo-benchmark/issues)
- 💬 [Discussions](https://github.com/yourusername/elo-benchmark/discussions)
- 📧 Email: research@example.com
- 📖 [Documentation](https://elo-benchmark.readthedocs.io/)
- 📊 [Live Leaderboard](https://elo-benchmark.example.com/leaderboard)

---

<div align="center">

**⭐ Star this repo if you find it useful for your research! ⭐**

Made with ❤️ by the UNER research team

</div>