import time
import uuid
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import logging

# Import sentence transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_MATCHING_AVAILABLE = True
except ImportError:
    SEMANTIC_MATCHING_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

from database import db
from models import LLMModel
from model_definitions import MODEL_CAPS
from logger_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Initialize sentence transformer model for semantic matching if available
semantic_model = None
def get_semantic_model():
    """Lazily load the sentence transformer model when needed."""
    global semantic_model, SEMANTIC_MATCHING_AVAILABLE
    if semantic_model is None and SEMANTIC_MATCHING_AVAILABLE:
        try:
            logger.info("Loading sentence transformer model for semantic matching")
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            SEMANTIC_MATCHING_AVAILABLE = False
    return semantic_model

def semantic_judge_vote_matching(judge_response: str, model_a_name: str, model_b_name: str) -> Tuple[str, float]:
    """
    Use semantic matching to determine which model the judge voted for.
    
    Args:
        judge_response: The judge's response text
        model_a_name: Name of the first model
        model_b_name: Name of the second model
        
    Returns:
        Tuple of (vote, confidence) where vote is 'a', 'b', or 'tie'
    """
    model = get_semantic_model()
    if not model:
        logger.warning("Semantic matching not available, falling back to text-based matching")
        return None, 0.0
    
    # Use short model names for more natural sentences
    model_a_short = model_a_name.split(" ")[0] if " " in model_a_name else model_a_name 
    model_b_short = model_b_name.split(" ")[0] if " " in model_b_name else model_b_name
    
    # Define reference phrases to compare against
    references = [
        # Generic references for Response A (don't mention model names)
        f"I choose Response A as the better response",
        f"Response A is superior to Response B",
        f"Response A demonstrates better clinical competence",
        f"I vote for Response A",
        f"My decision is Response A",
        
        # Generic references for Response B (don't mention model names)
        f"I choose Response B as the better response",
        f"Response B is superior to Response A",
        f"Response B demonstrates better clinical competence",
        f"I vote for Response B",
        f"My decision is Response B",
        
        # Generic references for tie
        f"Both responses are equally good",
        f"This is a tie between Response A and Response B",
        f"Neither response is clearly superior",
        f"I cannot definitively choose between Response A and Response B",
        f"My decision is a tie"
    ]
    
    # Extract last 5 sentences from the judge's response
    sentences = judge_response.replace('\n', ' ').split('.')
    last_sentences = ' '.join(sentences[-5:])
    
    # Look for direct mentions of model names in response (but only in last part to avoid false positives)
    model_a_mentioned = model_a_name.lower() in last_sentences.lower() or model_a_short.lower() in last_sentences.lower()
    model_b_mentioned = model_b_name.lower() in last_sentences.lower() or model_b_short.lower() in last_sentences.lower()
    
    if model_a_mentioned and not model_b_mentioned:
        logger.info(f"Direct mention of {model_a_name} found in judge response")
        return 'a', 0.9  # High confidence since model was directly mentioned
    elif model_b_mentioned and not model_a_mentioned:
        logger.info(f"Direct mention of {model_b_name} found in judge response")
        return 'b', 0.9  # High confidence since model was directly mentioned
    
    # Encode the references and the judge's response
    reference_embeddings = model.encode(references)
    response_embedding = model.encode(last_sentences)
    
    # Calculate cosine similarities
    similarities = util.cos_sim(response_embedding, reference_embeddings)[0]
    
    # Group similarities by vote type
    a_similarities = similarities[0:5].tolist()
    b_similarities = similarities[5:10].tolist()
    tie_similarities = similarities[10:15].tolist()
    
    # Get max similarity for each group
    max_a = max(a_similarities)
    max_b = max(b_similarities)
    max_tie = max(tie_similarities)
    
    # Get indices of max similarities for logging
    max_a_idx = a_similarities.index(max_a)
    max_b_idx = b_similarities.index(max_b)
    max_tie_idx = tie_similarities.index(max_tie)
    
    logger.info(f"Top semantic match for A: '{references[max_a_idx]}' (score: {max_a:.4f})")
    logger.info(f"Top semantic match for B: '{references[max_b_idx + 5]}' (score: {max_b:.4f})")
    logger.info(f"Top semantic match for tie: '{references[max_tie_idx + 10]}' (score: {max_tie:.4f})")
    
    # Determine the vote based on highest similarity
    if max_a > max_b and max_a > max_tie:
        return 'a', float(max_a)
    elif max_b > max_a and max_b > max_tie:
        return 'b', float(max_b)
    else:
        return 'tie', float(max_tie)

class Match:
    """
    Class representing a match between two LLM models.
    """
    def __init__(self, 
                match_id: Optional[str] = None,
                model_a: Optional[LLMModel] = None, 
                model_b: Optional[LLMModel] = None,
                timestamp: Optional[str] = None):
        """
        Initialize a new match.
        
        Args:
            match_id: Optional unique identifier, generated if not provided
            model_a: First competing model
            model_b: Second competing model
            timestamp: Optional timestamp, current time if not provided
        """
        logger.info(f"Creating new match")
        # Basic match info
        self.match_id = match_id or f"match_{str(uuid.uuid4())[:12]}"
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"
        
        # Participants
        self.participants = {
            "model_a": model_a.model_id if model_a else None,
            "model_b": model_b.model_id if model_b else None
        }
        
        logger.info(f"Match ID: {self.match_id}")
        logger.info(f"Participants: {self.participants}")
        
        # Prompt info
        self.prompt = {
            "case_text": "",
            "case_generator_id": "",
            "case_generator_elo": 0,
            "question_text": "",
            "question_generator_id": "",
            "question_generator_elo": 0
        }
        
        # Initialize response tracking
        self.responses = {}
        
        if model_a and model_b:
            self.participants["model_a"] = model_a.model_id
            self.participants["model_b"] = model_b.model_id
            
            self.responses[model_a.model_id] = {
                "text": "",
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "input_cost_usd": 0,
                "output_cost_usd": 0,
                "elo_raw_before": model_a.elo["raw"]["current"],
                "elo_cost_before": model_a.elo["cost_adjusted"]["current"],
                "elo_raw_after": model_a.elo["raw"]["current"],
                "elo_cost_after": model_a.elo["cost_adjusted"]["current"]
            }
            
            self.responses[model_b.model_id] = {
                "text": "",
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "input_cost_usd": 0,
                "output_cost_usd": 0,
                "elo_raw_before": model_b.elo["raw"]["current"],
                "elo_cost_before": model_b.elo["cost_adjusted"]["current"],
                "elo_raw_after": model_b.elo["raw"]["current"],
                "elo_cost_after": model_b.elo["cost_adjusted"]["current"]
            }
        
        # Initialize judgment structure
        self.judgment = {
            "raw_score": {},
            "cost_adjusted_score": {},
            "judges": []
        }
        
        # Metadata
        self.meta = {
            "case_id": f"case_{str(uuid.uuid4())[:12]}",
            "question_id": f"q_{str(uuid.uuid4())[:12]}",
            "stratum": "",
            "matchup_index": 0,
            "was_reused_case": False,
            "judgment_method": "elo-weighted softmax",
            "cost_temperature": 0.05,
            "elo_k_factor": 16
        }
    
    def set_prompt(self, case_text: str, question_text: str, 
                  case_generator: LLMModel, question_generator: LLMModel):
        """Set the prompt for the match."""
        logger.info(f"Setting prompt for match {self.match_id}")
        logger.info(f"Case generator: {case_generator.name}")
        logger.info(f"Question generator: {question_generator.name}")
        
        self.prompt = {
            "case_text": case_text,
            "case_generator_id": case_generator.model_id,
            "case_generator_elo": case_generator.elo["raw"]["current"],
            "question_text": question_text,
            "question_generator_id": question_generator.model_id,
            "question_generator_elo": question_generator.elo["raw"]["current"]
        }
        
        # Add this match to the generators' record
        case_generator.match_ids["cases_generated"].append(self.match_id)
        question_generator.match_ids["questions_generated"].append(self.match_id)
        
        # Save generators to DB
        case_generator.save_to_db()
        question_generator.save_to_db()
        
        return self
    
    def set_stratum(self, stratum: str, matchup_index: int):
        """Set the ELO stratum for the match."""
        logger.info(f"Setting ELO stratum {stratum} for match {self.match_id}")
        self.meta["stratum"] = stratum
        self.meta["matchup_index"] = matchup_index
        return self
    
    def collect_response(self, model: LLMModel, response_text: str, tokens: int, cost_usd: float, 
                         input_tokens: int = None, output_tokens: int = None,
                         input_cost: float = None, output_cost: float = None):
        """Record a model's response with detailed token usage."""
        logger.info(f"Collecting response from {model.name} for match {self.match_id}")
        logger.info(f"Tokens used: {tokens}, Cost: ${cost_usd:.4f}")
        
        if input_tokens is None:
            # Estimate input/output tokens if not provided
            input_tokens = tokens // 3  # Rough estimate, adjust as needed
            output_tokens = tokens - input_tokens
        
        if input_cost is None:
            # Estimate input/output costs if not provided
            input_cost = cost_usd * (input_tokens / tokens) if tokens > 0 else 0
            output_cost = cost_usd * (output_tokens / tokens) if tokens > 0 else 0
        
        logger.info(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        logger.info(f"Input cost: ${input_cost:.4f}, Output cost: ${output_cost:.4f}")
        
        self.responses[model.model_id] = {
            "text": response_text,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "elo_raw_before": model.elo["raw"]["current"],
            "elo_cost_before": model.elo["cost_adjusted"]["current"],
            "elo_raw_after": model.elo["raw"]["current"],
            "elo_cost_after": model.elo["cost_adjusted"]["current"]
        }
        return self
    
    def add_judge(self, judge: LLMModel, vote: str, weight: float, vote_type: str = "win",
                 prompt: str = None, response: str = None, tokens: int = 0, cost: float = 0.0):
        """
        Add a judge's vote.
        
        Args:
            judge: The judge model
            vote: The model ID that the judge voted for, or "draw" if judge declared a draw
            weight: The judge's weight (will be updated later based on ELO)
            vote_type: Either "win" for a vote for one model, or "draw" for a tie
            prompt: The prompt sent to the judge
            response: The full response from the judge
            tokens: Number of tokens used by the judge
            cost: Cost of the judge's evaluation
        """
        logger.info(f"Adding judge {judge.name} to match {self.match_id}")
        logger.info(f"Vote: {vote}, Vote type: {vote_type}, Weight: {weight:.2f}")
        
        judge_entry = {
            "judge_id": judge.model_id,
            "judge_name": judge.name,
            "elo_at_judgment": judge.elo["raw"]["current"],
            "vote": vote,
            "weight": weight,
            "vote_type": vote_type  # "win" or "draw"
        }
        
        # Add detailed information if provided
        if prompt is not None:
            judge_entry["prompt"] = prompt
        if response is not None:
            judge_entry["response"] = response
        if tokens > 0:
            judge_entry["tokens"] = tokens
        if cost > 0:
            judge_entry["cost"] = cost
            
        self.judgment["judges"].append(judge_entry)
        
        # Add this match to the judge's record
        judge.match_ids["judged"].append(self.match_id)
        judge.save_to_db()
        
        return self
    
    def set_scores(self, raw_scores: Dict[str, float], cost_adjusted_scores: Dict[str, float]):
        """Set the final scores."""
        logger.info(f"Setting scores for match {self.match_id}")
        logger.info(f"Raw scores: {raw_scores}")
        logger.info(f"Cost-adjusted scores: {cost_adjusted_scores}")
        
        self.judgment["raw_score"] = raw_scores
        self.judgment["cost_adjusted_score"] = cost_adjusted_scores
        return self
    
    def update_elo(self, model_a: LLMModel, model_b: LLMModel):
        """Record ELO ratings for both models in the match data without updating them."""
        logger.info(f"Recording ELO ratings for match {self.match_id}")
        
        model_a_id = model_a.model_id
        model_b_id = model_b.model_id
        
        # Record current ELO values
        self.responses[model_a_id]["elo_raw_after"] = model_a.elo["raw"]["current"]
        self.responses[model_a_id]["elo_cost_after"] = model_a.elo["cost_adjusted"]["current"]
        self.responses[model_b_id]["elo_raw_after"] = model_b.elo["raw"]["current"]
        self.responses[model_b_id]["elo_cost_after"] = model_b.elo["cost_adjusted"]["current"]
        
        # Create a nice table for the ELO update in logs
        header = "┌─────────────────────┬──────────┬──────────┬──────────┐"
        logger.table(header)
        logger.table(f"│ {'Model':<19} │ {'Raw ELO':<8} │ {'Cost ELO':<8} │ {'Matches':<8} │")
        separator = "├─────────────────────┼──────────┼──────────┼──────────┤"
        logger.table(separator)
        logger.table(f"│ {model_a.name:<19} │ {model_a.elo['raw']['current']:<8.1f} │ {model_a.elo['cost_adjusted']['current']:<8.1f} │ {model_a.performance['total_matches_played']:<8} │")
        logger.table(f"│ {model_b.name:<19} │ {model_b.elo['raw']['current']:<8.1f} │ {model_b.elo['cost_adjusted']['current']:<8.1f} │ {model_b.performance['total_matches_played']:<8} │")
        bottom = "└─────────────────────┴──────────┴──────────┴──────────┘"
        logger.table(bottom)
        
        logger.info(f"Recorded ELO - {model_a.name}: {model_a.elo['raw']['current']:.1f}, {model_b.name}: {model_b.elo['raw']['current']:.1f}")
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "_id": self.match_id,
            "timestamp": self.timestamp,
            "participants": self.participants,
            "prompt": self.prompt,
            "responses": self.responses,
            "judgment": self.judgment,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create match object from dictionary."""
        match = cls(
            match_id=data.get("_id"),
            timestamp=data.get("timestamp")
        )
        
        # Set fields directly from data
        match.participants = data.get("participants", {})
        match.prompt = data.get("prompt", {})
        match.responses = data.get("responses", {})
        match.judgment = data.get("judgment", {"raw_score": {}, "cost_adjusted_score": {}, "judges": []})
        match.meta = data.get("meta", {})
        
        return match
    
    def save_to_db(self):
        """Save match to database."""
        return db.save_match(self.to_dict())
    
    @classmethod
    def load_from_db(cls, match_id):
        """Load match from database by ID."""
        match_data = db.get_match(match_id)
        if match_data:
            return cls.from_dict(match_data)
        return None
    
    @classmethod
    def get_all_matches(cls):
        """Get all matches from the database."""
        matches_data = db.get_matches({})
        return [cls.from_dict(match_data) for match_data in matches_data]
    
    def __str__(self):
        model_a = list(self.responses.keys())[0] if self.responses else "TBD"
        model_b = list(self.responses.keys())[1] if len(self.responses) > 1 else "TBD"
        
        if "raw_score" in self.judgment and model_a in self.judgment["raw_score"]:
            score_a = self.judgment["raw_score"][model_a]
            score_b = self.judgment["raw_score"][model_b]
            return f"Match {self.match_id}: {model_a} vs {model_b} ({score_a:.2f} - {score_b:.2f})"
        else:
            return f"Match {self.match_id}: {model_a} vs {model_b} (Pending)"


def generate_case_prompt():
    """Template for prompting case-generation LLM."""
    prompt = (
        "You are a distinguished mathematician creating a mathematical reasoning scenario for advanced evaluation. Your task is to present ONLY the mathematical facts and problem setup as they would appear in a rigorous mathematical context."

        "CRITICAL: Present ONLY the raw mathematical data and problem statement. DO NOT include:"
        "- Any solutions or solution methods"
        "- Mathematical analysis or interpretation"
        "- Step-by-step reasoning"
        "- Questions about the problem"
        "- Explanations of mathematical concepts"
        "- Conclusions or final answers"

        "Format your response as a mathematical problem with these sections:"
        "Problem Context & Domain"
        "Given Information"
        "Mathematical Objects & Definitions"
        "Constraints & Conditions"
        "Numerical Data"
        "Geometric Configuration (if applicable)"
        "Functional Relationships"
        "Boundary Conditions"
        "Measurement Results"

        "Requirements:"
        "- Create a complex mathematical scenario requiring deep reasoning"
        "- Include multiple mathematical approaches without stating them"
        "- Provide comprehensive mathematical data without interpretation"
        "- End immediately after presenting the final piece of given information"

        "Example format: \"The function f(x) = x³ - 3x² + 2x - 1 is defined on the interval [0, 4] with f'(2) = -1.\" NOT \"The function shows a local extremum indicating...\""

        "STOP writing immediately after presenting all mathematical data. Do not add any analysis, questions, or commentary."
    )
    logger.info(f"Generated mathematical reasoning case prompt with updated format")
    logger.info(f"Prompt: '{prompt[:100]}...'")
    return prompt

def generate_question_prompt(case_text: str):
    """Template for prompting question-generation LLM."""
    prompt = (
        f"Mathematical Problem:\n{case_text}\n\n"
        "Based on the above mathematical problem, your task is to formulate ONE precise, challenging question for rigorous mathematical evaluation."
        
        "CRITICAL CONSTRAINTS:"
        "- Generate ONLY the question - DO NOT provide any answer, solution, or reasoning"
        "- This is a theoretical mathematical evaluation, NOT a computational exercise"
        "- Questions must be answerable through mathematical analysis of the provided problem data"
        "- DO NOT ask for lengthy calculations or extensive numerical computations"
        
        "Question requirements - PRECISE MATHEMATICAL FOCUS:"
        "- Focus on mathematical reasoning, proof techniques, and analytical thinking"
        "- Demand synthesis of mathematical concepts with rigorous logical deduction"
        "- Have one clear mathematically optimal answer based on mathematical principles"
        "- Test mathematical insight, not computational ability or memorization"
        "- Require deep understanding of mathematical relationships in the problem"
        
        "Appropriate precise mathematical question examples:"
        "- \"What is the nature of the critical points of this function on the given domain?\""
        "- \"What mathematical theorem or principle directly applies to solve this problem?\""
        "- \"What is the optimal mathematical approach to establish the required result?\""
        "- \"Which mathematical property is most crucial for determining the solution?\""
        "- \"What is the underlying mathematical structure that governs this problem?\""
        "- \"What constraint is mathematically most restrictive in this scenario?\""
        "- \"What mathematical transformation would simplify this problem most effectively?\""
        
        "STOP immediately after writing the question. Do not provide:"
        "- Answer choices (unless specifically creating multiple choice)"
        "- Explanations of the answer"
        "- Reasoning behind the question"
        "- Additional commentary or mathematical insights"
        
        "Format: Write one clear, mathematically precise question ending with a question mark. Then STOP."
    )
    logger.info(f"Generated mathematical reasoning question prompt for problem of length {len(case_text)} characters")
    logger.info(f"Using updated question format focusing on precise mathematical evaluation")
    return prompt

def generate_answer_prompt(case_text: str, question_text: str):
    """Template for prompting player models to answer mathematical reasoning questions."""
    prompt = (
        f"Mathematical Problem:\n{case_text}\n\n"
        f"Question:\n{question_text}\n\n"
        "You are an expert mathematician providing your best answer to the above mathematical question. Demonstrate your mathematical reasoning and knowledge to provide the most rigorous and appropriate response."
        
        "Instructions:"
        "- Answer the question directly and with mathematical precision"
        "- Provide clear mathematical reasoning to support your answer"
        "- Consider relevant alternative mathematical approaches where appropriate"
        "- Base your response on established mathematical principles and theorems"
        "- Be concise but thorough - avoid unnecessary mathematical elaboration"
        "- If multiple approaches are valid, choose the most mathematically elegant and rigorous option"
        "- Acknowledge uncertainty only when genuinely present in the mathematical scenario"
        
        "Format your response:"
        "1. Direct answer to the question"
        "2. Brief mathematical reasoning (2-3 sentences maximum)"
        "3. Key supporting mathematical evidence from the problem"
        
        "Focus on demonstrating mathematical competence and sound mathematical judgment. Your response will be evaluated against another expert's answer."
    )
    logger.info(f"Generated mathematical reasoning answer prompt for problem length {len(case_text)} and question length {len(question_text)}")
    return prompt

def generate_judge_prompt(case_text: str, question_text: str, first_response: str, second_response: str):
    """Template for prompting judge models to evaluate responses."""
    prompt = (
        f"Mathematical Problem:\n{case_text}\n\n"
        f"Question:\n{question_text}\n\n"
        f"Response A:\n{first_response}\n\n"
        f"Response B:\n{second_response}\n\n"
        "You are an expert mathematical reviewer evaluating two anonymous responses to the above mathematical question. Your role is to determine which response demonstrates superior mathematical reasoning competence."
        
        "Evaluation framework (weighted criteria):"
        
        "PRIMARY (70% weight):"
        "- Mathematical accuracy of solutions/reasoning"
        "- Appropriateness of mathematical methods and theorem application"
        "- Rigor and logical validity of mathematical arguments"
        
        "SECONDARY (30% weight):"
        "- Completeness without irrelevant mathematical details"
        "- Recognition of mathematical nuances and edge cases"
        "- Appropriate confidence level and acknowledgment of mathematical uncertainty"
        
        "Evaluation process:"
        "1. Identify any factual mathematical errors (automatic disqualification if fundamentally incorrect)"
        "2. Assess quality of mathematical reasoning and logical deduction"
        "3. Evaluate appropriateness of mathematical methods for this specific problem"
        "4. Consider which response shows deeper understanding of mathematical principles"
        
        "Decision rules:"
        "- Choose Response A, Response B, or Tie"
        "- Ties only when responses are genuinely equivalent in mathematical merit"
        "- Shorter, more precise answers can be superior to longer, rambling ones"
        "- Prioritize mathematical rigor and appropriate mathematical conservatism"
        "- Consider: \"Which response would I trust more for mathematical correctness?\""
        
        "Focus exclusively on mathematical competence. Ignore formatting, writing style, or response structure unless it affects mathematical clarity."
        
        "IMPORTANT: After your analysis, you MUST end your evaluation with a clear and explicit verdict using exactly one of these three formats:"
        
        "VERDICT: Response A is superior."
        "VERDICT: Response B is superior."
        "VERDICT: This is a tie."
        
        "IMPORTANT: If your verdict is not provided in exactly one of these three formats, your judgment will be considered invalid and automatically discarded. Judges who fail to follow this format will be removed from the judge pool and may incur an ELO rating penalty."
        
        "Examples of properly formatted verdicts:"
        "Example 1: 'After evaluating both responses, Response A shows better mathematical accuracy and clearer reasoning. VERDICT: Response A is superior.'"
        "Example 2: 'While both responses identify key concepts, Response B provides better mathematical rigor and theorem application. VERDICT: Response B is superior.'"
        "Example 3: 'Both responses demonstrate equivalent mathematical competence and reasoning. VERDICT: This is a tie.'"
        
        "Your evaluation MUST end with the exact verdict format as shown above."
    )
    
    logger.info(f"Generated mathematical reasoning judge evaluation prompt with standardized verdict format")
    return prompt

# Pairing and Tournament management functions
def calculate_elo_strata(models: List[LLMModel], strata_size: int = 100) -> Dict[str, List[LLMModel]]:
    """Group models into ELO strata for fairer matchups."""
    logger.info(f"Calculating ELO strata with stratum size {strata_size}")
    strata = {}
    
    for model in models:
        elo = int(model.elo["raw"]["current"])
        stratum_base = (elo // strata_size) * strata_size
        stratum_key = f"{stratum_base}-{stratum_base + strata_size}"
        
        if stratum_key not in strata:
            strata[stratum_key] = []
        
        strata[stratum_key].append(model)
    
    # Log the strata distributions
    logger.info(f"Created {len(strata)} ELO strata:")
    for stratum_key, models_in_stratum in strata.items():
        logger.info(f"  {stratum_key}: {len(models_in_stratum)} models - {', '.join(m.name for m in models_in_stratum)}")
    
    return strata


def generate_swiss_pairings(models: List[LLMModel], 
                          delta: int = 100,
                          prior_matches: Optional[Set[Tuple[str, str]]] = None) -> List[Tuple[LLMModel, LLMModel]]:
    """
    Generate Swiss-style pairings.
    
    Args:
        models: List of LLMModel objects
        delta: Maximum ELO difference for a valid pairing
        prior_matches: Set of tuples (model_a_name, model_b_name) representing previous matches
        
    Returns:
        List of (model_a, model_b) tuples for matches
    """
    logger.info(f"Generating Swiss pairings with {len(models)} models")
    logger.info(f"Maximum ELO delta: {delta}, Prior matches: {len(prior_matches or set())}")
    
    if prior_matches is None:
        prior_matches = set()
    
    # Sort models by current ELO
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    logger.info(f"Models sorted by ELO: {', '.join([m.name for m in sorted_models])}")
    
    # Track match count for each model
    match_counts = {model.name: 0 for model in models}
    
    # Track used models
    used_models = set()
    
    # Generate pairings
    pairings = []
    
    for i, model_a in enumerate(sorted_models):
        if model_a.name in used_models or match_counts[model_a.name] >= MODEL_CAPS["max_matches_per_model"]:
            continue
            
        logger.info(f"Finding opponent for {model_a.name} (ELO: {model_a.elo['raw']['current']:.1f})")
        
        # Look for a valid opponent
        for j in range(i+1, len(sorted_models)):
            model_b = sorted_models[j]
            
            # Skip if already used or too many matches
            if (model_b.name in used_models or 
                match_counts[model_b.name] >= MODEL_CAPS["max_matches_per_model"]):
                continue
                
            # Check ELO difference
            elo_diff = abs(model_a.elo["raw"]["current"] - model_b.elo["raw"]["current"])
            logger.info(f"  Potential opponent: {model_b.name} (ELO: {model_b.elo['raw']['current']:.1f}, diff: {elo_diff:.1f})")
            
            if elo_diff > delta:
                logger.info(f"  Rejected: ELO difference {elo_diff:.1f} > {delta}")
                continue
                
            # Check if they've already played
            pair_key = tuple(sorted([model_a.name, model_b.name]))
            if pair_key in prior_matches:
                logger.info(f"  Rejected: These models have already played each other")
                continue
                
            # Valid pairing found
            logger.info(f"  Valid pairing found: {model_a.name} vs {model_b.name}")
            pairings.append((model_a, model_b))
            used_models.add(model_a.name)
            used_models.add(model_b.name)
            match_counts[model_a.name] += 1
            match_counts[model_b.name] += 1
            prior_matches.add(pair_key)
            break
    
    logger.info(f"Generated {len(pairings)} pairings")
    for i, (model_a, model_b) in enumerate(pairings, 1):
        logger.info(f"Pairing {i}: {model_a.name} vs {model_b.name}")
    
    return pairings


def select_judges(all_models: List[LLMModel], 
                participants: List[str], 
                count: int = 3) -> List[LLMModel]:
    """
    Select judge models that aren't participating in the match.
    
    Args:
        all_models: List of all available models
        participants: List of model names that are in the match
        count: Number of judges to select
        
    Returns:
        List of selected judge models
    """
    logger.info(f"Selecting {count} judges, excluding participants: {', '.join(participants)}")
    
    # Filter out participant models
    eligible_judges = [m for m in all_models if m.name not in participants]
    logger.info(f"Found {len(eligible_judges)} eligible judge candidates")
    
    # Sort by number of times they've judged (to balance judging duties)
    eligible_judges.sort(key=lambda m: len(m.match_ids["judged"]))
    
    # Take the least-used judges first
    selected_judges = eligible_judges[:count]
    logger.info(f"Selected judges: {', '.join(j.name for j in selected_judges)}")
    
    # Avoid nested quotes in f-string by preparing the list separately
    judge_counts = []
    for j in selected_judges:
        judge_counts.append(f"{j.name}:{len(j.match_ids['judged'])}")
    logger.info(f"Judge judging counts: {', '.join(judge_counts)}")
    
    return selected_judges


def softmax(x, tau=1.0):
    """Apply softmax function with temperature."""
    logger.debug(f"Applying softmax with temperature {tau} to {len(x)} values")
    x_scaled = np.array(x) / tau
    e_x = np.exp(x_scaled - np.max(x_scaled))
    result = e_x / e_x.sum()
    logger.debug(f"Softmax input range: {min(x):.1f}-{max(x):.1f}, output range: {min(result):.4f}-{max(result):.4f}")
    return result


def weight_votes(judges: List[Dict[str, Any]], temperature: float = 300) -> Dict[str, float]:
    """
    Weight judges' votes by their raw ELO ratings using softmax.
    
    Args:
        judges: List of judge data from Match.judgment["judges"]
        temperature: Temperature for softmax (higher = more equal weighting)
        
    Returns:
        Dictionary mapping model_id to weighted vote score
    """
    logger.info(f"Weighting {len(judges)} judge votes with softmax temperature {temperature}")
    logger.info(f"Using raw ELO for judge weighting: w_k = e^(R_k^raw/τ) / Σ_j=1^J e^(R_j^raw/τ)")
    
    # Extract ELO ratings and votes
    elos = [judge["elo_at_judgment"] for judge in judges]
    votes = [judge["vote"] for judge in judges]
    
    logger.info(f"Judge ELOs: {elos}")
    logger.info(f"Judge votes: {votes}")
    
    # Calculate weights using softmax
    weights = softmax(elos, temperature)
    
    # Log the weights
    for i, (elo, weight) in enumerate(zip(elos, weights)):
        logger.info(f"Judge {i+1}: ELO={elo:.1f}, weight={weight:.4f}")
    
    # Tally weighted votes
    scores = {}
    for vote, weight in zip(votes, weights):
        if vote not in scores:
            scores[vote] = 0
        scores[vote] += weight
    
    logger.info(f"Weighted vote scores: {scores}")
    return scores


def calculate_cost_adjusted_score(raw_scores: Dict[str, float], 
                                model_responses: Dict[str, Dict[str, Any]],
                                cost_temperature: float = 0.05) -> Dict[str, float]:
    """
    Calculate cost-adjusted scores based on model response costs.
    
    Args:
        raw_scores: Dictionary mapping model_id to raw score
        model_responses: Dictionary of model responses with cost info
        cost_temperature: How much to adjust for cost (higher = more adjustment)
        
    Returns:
        Dictionary mapping model_id to cost-adjusted score
    """
    logger.info(f"Calculating cost-adjusted scores (Section 3.7)")
    logger.info(f"Using cost temperature τ_c={cost_temperature}")
    
    if len(raw_scores) != 2:
        logger.warning(f"Cost adjustment requires exactly two models, got {len(raw_scores)}")
        return raw_scores
    
    model_ids = list(raw_scores.keys())
    model_a, model_b = model_ids[0], model_ids[1]
    
    # Get costs
    cost_a = model_responses[model_a]["cost_usd"]
    cost_b = model_responses[model_b]["cost_usd"]
    
    logger.info(f"Model costs: {model_a}=${cost_a:.6f}, {model_b}=${cost_b:.6f}")
    
    # Skip adjustment if costs are zero or missing
    if cost_a <= 0 or cost_b <= 0:
        logger.warning("Missing or zero costs, skipping cost adjustment")
        return raw_scores.copy()
    
    # Calculate efficiency factors (Equation 7)
    eff_a = np.exp(-cost_a / cost_temperature)
    eff_b = np.exp(-cost_b / cost_temperature)
    
    logger.info(f"Efficiency factors (Equation 7): {model_a}={eff_a:.6f}, {model_b}={eff_b:.6f}")
    
    # Calculate cost-adjusted scores (Equations 8-9)
    numerator_a = raw_scores[model_a] * eff_a
    numerator_b = raw_scores[model_b] * eff_b
    denominator = numerator_a + numerator_b
    
    adj_score_a = numerator_a / denominator
    adj_score_b = numerator_b / denominator
    
    logger.info(f"Cost-adjusted scores calculated using Equations 8-9:")
    logger.info(f"  S_A^adj = (S_A^raw · eff_A) / (S_A^raw · eff_A + S_B^raw · eff_B)")
    logger.info(f"  {model_a}: {raw_scores[model_a]:.4f} → {adj_score_a:.4f}")
    logger.info(f"  {model_b}: {raw_scores[model_b]:.4f} → {adj_score_b:.4f}")
    
    return {model_a: adj_score_a, model_b: adj_score_b}


# Example usage
if __name__ == "__main__":
    from models import LLMModel, initialize_models
    from model_definitions import MODELS
    
    # Initialize models
    all_models = initialize_models(MODELS)
    
    # Create a sample match
    match = Match(model_a=all_models[0], model_b=all_models[1])
    print(f"Created match: {match}")
    
    # Generate Swiss pairings 
    prior_matches = set()  # Initialize an empty set of prior matches
    pairings = generate_swiss_pairings(all_models, prior_matches=prior_matches)
    
    for model_a, model_b in pairings:
        print(f"Pairing: {model_a.name} vs {model_b.name}")