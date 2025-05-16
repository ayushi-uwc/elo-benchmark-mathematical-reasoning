# tournament.py
import random
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import logging
import math
import asyncio
from models import LLMModel
from matches import (
    Match, generate_case_prompt, generate_question_prompt, 
    generate_answer_prompt, generate_judge_prompt, 
    softmax, weight_votes, calculate_cost_adjusted_score,
    semantic_judge_vote_matching, SEMANTIC_MATCHING_AVAILABLE
)
from model_definitions import MODEL_CAPS
from database import db
from logger_config import get_logger

# Define the verdict pattern constant
VALID_VERDICT_PATTERN = r"VERDICT:\s*(Response\s*A\s*is\s*superior|Response\s*B\s*is\s*superior|This\s*is\s*a\s*tie)"

# Get logger for this module
logger = get_logger(__name__)

def get_top_performers(models: List[LLMModel], percentage: float = 0.25) -> List[LLMModel]:
    """Get the top percentage of models by ELO rating."""
    logger.info(f"Getting top {percentage*100}% performers by ELO rating")
    # Sort by raw ELO score
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    
    # Log ELO scores for all models
    logger.info("Current ELO scores for all models:")
    for model in sorted_models:
        logger.info(f"{model.name}: Raw ELO={model.elo['raw']['current']:.1f}, Cost-Adj ELO={model.elo['cost_adjusted']['current']:.1f}")
    
    # Take top percentage
    count = max(1, int(len(models) * percentage))
    top_models = sorted_models[:count]
    logger.info(f"Selected top {len(top_models)} models: {', '.join(m.name for m in top_models)}")
    return top_models

def get_nested_attr(obj, attr_path):
    """Access nested attributes or dictionary keys using dot notation."""
    for attr in attr_path.split('.'):
        if isinstance(obj, dict):
            # Handle dictionary access
            obj = obj.get(attr)
        else:
            # Handle object attribute access
            obj = getattr(obj, attr)
    return obj

def inverse_weighted_choice(models, match_count_key="performance.total_matches_played", max_matches=20):
    """
    Select one model from a list with inverse match-count weighting,
    excluding models with >= max_matches.
    """
    logger.info(f"\n{'#'*20} INVERSE WEIGHTED SELECTION {'#'*20}")
    logger.info(f"Running inverse_weighted_choice with {len(models)} models")
    logger.info(f"Match count key: {match_count_key}, Max matches: {max_matches}")
    
    logger.info("\nFILTERING ELIGIBLE MODELS:")
    logger.info(f"Filtering out models with >= {max_matches} matches")
    
    # Log all models before filtering
    logger.info(f"\nAll models before filtering ({len(models)}):")
    for m in models:
        match_count = get_nested_attr(m, match_count_key)
        status = "ELIGIBLE" if match_count < max_matches else "EXCLUDED (match cap)"
        logger.info(f"  {m.name}: {match_count} matches - {status}")
    
    eligible = [
        m for m in models
        if get_nested_attr(m, match_count_key) < max_matches
    ]
    
    logger.info(f"Found {len(eligible)}/{len(models)} eligible models under match limit")
    if len(eligible) < len(models):
        excluded = [m for m in models if m not in eligible]
        logger.info(f"Excluded models: {', '.join(m.name for m in excluded)}")
    
    if not eligible:
        logger.warning("No eligible models found - cannot make selection")
        return None
    
    logger.info("\nCALCULATING INVERSE MATCH-COUNT WEIGHTS (Equation 1):")
    logger.info("w_i = 1/(1+n_i) where n_i = number of matches for model m_i")
    
    # Calculate weights
    model_weights = []
    for m in eligible:
        match_count = get_nested_attr(m, match_count_key)
        weight = 1 / (1 + match_count)
        model_weights.append((m, match_count, weight))
    
    # Sort by weight for clearer display
    model_weights.sort(key=lambda x: -x[2])  # Sort by weight descending
    
    # Display all weights
    for m, matches, weight in model_weights:
        logger.info(f"  {m.name}: matches={matches}, w_{m.name} = 1/(1+{matches}) = {weight:.6f}")
    
    # Calculate probabilities
    models_only = [m for m, _, _ in model_weights]
    weights_only = [w for _, _, w in model_weights]
    total_weight = sum(weights_only)
    
    logger.info(f"\nTotal weight (sum of all weights): {total_weight:.6f}")
    
    logger.info("\nCALCULATING SELECTION PROBABILITIES (Equation 2):")
    logger.info("P(m_i) = w_i / Σ_j w_j")
    
    probs = [w / total_weight for w in weights_only]
    
    for i, (m, matches, weight) in enumerate(zip(models_only, weights_only, probs)):
        prob = probs[i]
        logger.info(f"  P({m.name}) = {weight:.6f}/{total_weight:.6f} = {prob:.6f} ({prob*100:.2f}%)")
    
    # Make random selection based on weights
    logger.info("\nMAKING CATEGORICAL RANDOM SELECTION (Equation 3):")
    logger.info("m ~ Categorical(P(m_1), P(m_2), ..., P(m_N))")
    
    selected = random.choices(models_only, weights=weights_only, k=1)[0]
    
    logger.info(f"Selected model: {selected.name} (probability: {probs[models_only.index(selected)]:.6f})")
    logger.info(f"{'#'*60}\n")
    
    return selected

def select_pair(models, initial_delta=50, delta_step=25, max_delta=400, max_matches=20,
                elo_key="elo.cost_adjusted.current", match_count_key="performance.total_matches_played", 
                prior_matches=None):
    """
    Select two models for a match following the paper's algorithm:
    1. Player 1 (m_A): Selected with probability inversely proportional to match count (Eq. 1-2)
    2. Player 2 (m_B): Selected from ELO stratum S_Δ(m_A) using inverse match-count weighting (Eq. 4-7)
    
    Args:
        models: List of LLMModel objects
        initial_delta: Starting ELO difference tolerance (Δ in Eq. 4)
        delta_step: How much to increase delta if no pairs found
        max_delta: Maximum allowed ELO difference
        max_matches: Maximum matches per model (n_max)
        elo_key: Path to the ELO rating to use
        match_count_key: Path to the match count attribute
        prior_matches: Set of tuples (model_a_name, model_b_name) representing previous matches
    """
    logger.info(f"\n{'='*30} PLAYER SELECTION {'='*30}")
    logger.info(f"Selecting pair from {len(models)} available models")
    logger.info(f"Parameters: initial_delta={initial_delta}, max_delta={max_delta}, max_matches={max_matches}")
    logger.info(f"Prior matches: {len(prior_matches) if prior_matches else 0}")
    
    # If models list is empty or has only one model, return None, None
    if not models or len(models) < 2:
        logger.warning("Not enough models available to form a pair.")
        return None, None
    
    # Log prior matches detail
    if prior_matches and len(prior_matches) > 0:
        logger.info("Prior match pairs:")
        for i, pair in enumerate(sorted(prior_matches), 1):
            logger.info(f"  {i}. {pair[0]} vs {pair[1]}")
    
    logger.info(f"Using ELO key: {elo_key}")
    logger.info(f"Using match count key: {match_count_key}")
    
    # Initialize prior_matches if None
    if prior_matches is None:
        prior_matches = set()
    
    # Check if any model is eligible for more matches
    eligible_models = [m for m in models if get_nested_attr(m, match_count_key) < max_matches]
    if len(eligible_models) < 2:
        logger.warning(f"Not enough models with fewer than {max_matches} matches.")
        return None, None
    
    # STEP 1: Select Player 1 using inverse match-count weighting (Equations 1-2)
    logger.info("\n--- STEP 1: SELECTING PLAYER 1 ---")
    logger.info("Using inverse match-count weighting (Equations 1-3)")
    player_1 = inverse_weighted_choice(models, match_count_key, max_matches)
    
    if player_1 is None:
        logger.warning("Failed to find eligible Player 1 - aborting pair selection")
        return None, None  # No eligible player found

    elo_1 = get_nested_attr(player_1, elo_key)
    logger.info(f"Selected Player 1: {player_1.name}")
    logger.info(f"  ELO: {elo_1:.1f}")
    logger.info(f"  Matches played: {get_nested_attr(player_1, match_count_key)}")
    
    # STEP 2: Find opponent from ELO stratum (Equations 4-7)
    logger.info("\n--- STEP 2: SELECTING PLAYER 2 FROM ELO STRATUM ---")
    logger.info("Following Equation (4): S_Δ(m_A) = {m_j ∈ M \\ {m_A} | |R_j^cost - R_A^cost| ≤ Δ}")
    logger.info(f"Player 1 ({player_1.name}) has ELO = {elo_1:.1f}")
    
    # Iterate through expanding delta thresholds until a valid stratum is found
    current_delta = initial_delta
    while current_delta <= max_delta:
        logger.info(f"\nTrying ELO delta = {current_delta}")
        
        # Equation 4: Define the stratum S_Δ(m_A)
        logger.info(f"Constructing stratum S_{current_delta}({player_1.name})")
        logger.info("Filtering for models:")
        logger.info(f"  1. Not {player_1.name}")
        logger.info(f"  2. Under {max_matches} matches")
        logger.info(f"  3. Within {current_delta} ELO points of {player_1.name}")
        
        # Log ELO range
        min_elo = elo_1 - current_delta
        max_elo = elo_1 + current_delta
        logger.info(f"ELO range: [{min_elo:.1f}, {max_elo:.1f}]")
        
        # Check all models - show details about why they might be excluded
        logger.info("\nEvaluating all models for inclusion in stratum:")
        
        stratum = []
        for m in models:
            m_elo = get_nested_attr(m, elo_key)
            m_matches = get_nested_attr(m, match_count_key)
            elo_diff = abs(m_elo - elo_1)
            
            if m.model_id == player_1.model_id:
                status = "EXCLUDED (same as Player 1)"
            elif m_matches >= max_matches:
                status = f"EXCLUDED (reached match cap: {m_matches} >= {max_matches})"
            elif elo_diff > current_delta:
                status = f"EXCLUDED (ELO diff too large: {elo_diff:.1f} > {current_delta})"
            else:
                status = "INCLUDED in stratum"
                stratum.append(m)
            
            logger.info(f"  {m.name}: ELO={m_elo:.1f}, diff={elo_diff:.1f}, matches={m_matches} - {status}")
        
        logger.info(f"\nFound {len(stratum)} models in stratum with delta = {current_delta}")
        
        if stratum:
            # Equation 5-7: Select Player 2 using inverse match-count weighting within the stratum
            logger.info("\nSelecting Player 2 from stratum using inverse match-count weighting")
            logger.info("Following Equations 5-7:")
            logger.info("  Equation 5: w_j = 1/(1+n_j) for each m_j in stratum")
            logger.info("  Equation 6: P(m_j|S_Δ) = w_j / Σ_{k∈S_Δ(m_A)} w_k")
            logger.info("  Equation 7: m_B ~ Categorical(P(m_j|S_Δ))_{j∈S_Δ(m_A)}")
            
            # Calculate weights based on inverse match count (Equation 5)
            weights = []
            models_in_stratum = []
            match_counts = []
            
            logger.info("\nInverse match-count weights within stratum:")
            for m in stratum:
                match_count = get_nested_attr(m, match_count_key)
                weight = 1.0 / (1.0 + match_count)
                
                models_in_stratum.append(m)
                match_counts.append(match_count)
                weights.append(weight)
                
                logger.info(f"  {m.name}: matches={match_count}, w_{m.name} = 1/(1+{match_count}) = {weight:.6f}")
            
            # Calculate normalized probabilities (Equation 6)
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            logger.info(f"\nTotal stratum weight: {total_weight:.6f}")
            logger.info("\nNormalized selection probabilities:")
            
            for i, (m, w, p) in enumerate(zip(models_in_stratum, weights, probs)):
                logger.info(f"  P({m.name}|S_{current_delta}) = {w:.6f}/{total_weight:.6f} = {p:.6f} ({p*100:.2f}%)")
            
            # Equation 7: Sample Player 2 from categorical distribution
            logger.info("\nMaking categorical random selection for Player 2:")
            player_2 = random.choices(models_in_stratum, weights=probs, k=1)[0]
            
            p2_elo = get_nested_attr(player_2, elo_key)
            p2_matches = get_nested_attr(player_2, match_count_key)
            elo_diff = abs(p2_elo - elo_1)
            
            logger.info(f"Selected Player 2: {player_2.name}")
            logger.info(f"  ELO: {p2_elo:.1f}")
            logger.info(f"  ELO difference with Player 1: {elo_diff:.1f}")
            logger.info(f"  Matches played: {p2_matches}")
            logger.info(f"  Selection probability: {probs[models_in_stratum.index(player_2)]:.6f}")
            
            logger.info(f"\nFINAL PAIRING: {player_1.name} vs {player_2.name}")
            logger.info(f"{'='*68}\n")
            
            return player_1, player_2
        
        logger.info(f"Empty stratum S_{current_delta}({player_1.name}) - need to expand delta")
        logger.info(f"Incrementally expanding delta: {current_delta} → {current_delta + delta_step}")
        current_delta += delta_step

    logger.warning(f"Could not find suitable opponent for {player_1.name} within max delta {max_delta}")
    logger.info(f"Returning Player 1 without opponent: {player_1.name}, None")
    logger.info(f"{'='*68}\n")
    
    return player_1, None  # Found Player 1, but no opponent

def choose_case_question_judges(models: List[LLMModel], 
                               player_1: LLMModel, 
                               player_2: LLMModel, 
                               max_judges: int = 5) -> Dict:
    """
    Selects case generator, question generator, and judges from top models.
    Uses either top 7 models or top 25% (whichever is higher).
    Excludes player_1 and player_2 from all roles.
    All selections are random within the selected top models.
    
    Args:
        models: List of all available LLMModel objects
        player_1: First player model
        player_2: Second player model
        max_judges: Maximum number of judges to select
        
    Returns:
        Dictionary with case_generator, question_generator, and judges
    """
    logger.info(f"\n{'='*30} ROLE ASSIGNMENT {'='*30}")
    logger.info(f"Selecting case generator, question generator, and judges for match")
    logger.info(f"Match participants: {player_1.name} vs {player_2.name}")
    logger.info(f"Maximum judges: {max_judges}")
    logger.info(f"Using raw ELO for selection")
    
    # Define sorting key based on raw ELO
    sort_key = lambda m: m.elo["raw"]["current"]
    score_field = lambda m: m.elo["raw"]["current"]
    score_label = "raw ELO"
    
    # Check if all models have the same score (initial state of tournament)
    initial_scores = [score_field(m) for m in models]
    avg_score = sum(initial_scores) / len(initial_scores)
    max_score = max(initial_scores)
    min_score = min(initial_scores)
    score_range = max_score - min_score
    
    logger.info(f"\nGlobal {score_label} statistics:")
    logger.info(f"  Number of models: {len(models)}")
    logger.info(f"  Average {score_label}: {avg_score:.1f}")
    logger.info(f"  Min {score_label}: {min_score:.1f}")
    logger.info(f"  Max {score_label}: {max_score:.1f}")
    logger.info(f"  {score_label} range: {score_range:.1f}")
    
    all_same_score = all(abs(score - initial_scores[0]) < 0.001 for score in initial_scores)
    
    # Handle initial tournament state where all models have the same score
    if all_same_score:
        logger.info(f"\n*** SPECIAL CASE: ALL MODELS HAVE SAME {score_label.upper()} ***")
        logger.info("This is likely the initial tournament state")
        logger.info("For initial state, randomly selecting from ALL eligible models")
        
        # Get all eligible models (excluding players)
        excluded_names = {player_1.name, player_2.name}
        logger.info(f"\nExcluding match participants: {', '.join(excluded_names)}")
        
        eligible_models = [m for m in models if m.name not in excluded_names]
        logger.info(f"Found {len(eligible_models)}/{len(models)} eligible models")
        
        if len(eligible_models) < 2:
            error_msg = "Not enough eligible models to assign roles"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Randomly select case generator
        logger.info("\n--- SELECTING CASE GENERATOR ---")
        logger.info(f"Randomly selecting one model from {len(eligible_models)} eligible models")
        case_generator = random.choice(eligible_models)
        logger.info(f"Selected case generator: {case_generator.name}")
        logger.info(f"  {score_label}: {score_field(case_generator):.1f}")
        logger.info(f"  Previous cases generated: {len(case_generator.match_ids['cases_generated'])}")
        
        # Remove case generator from eligible models
        eligible_models = [m for m in eligible_models if m.name != case_generator.name]
        
        # Randomly select question generator
        logger.info("\n--- SELECTING QUESTION GENERATOR ---")
        logger.info(f"Randomly selecting one model from {len(eligible_models)} remaining eligible models")
        question_generator = random.choice(eligible_models)
        logger.info(f"Selected question generator: {question_generator.name}")
        logger.info(f"  {score_label}: {score_field(question_generator):.1f}")
        logger.info(f"  Previous questions generated: {len(question_generator.match_ids['questions_generated'])}")
        
        # Remove question generator from eligible models
        eligible_models = [m for m in eligible_models if m.name != question_generator.name]
        
        # Randomly select judges (up to max_judges or all remaining models)
        logger.info("\n--- SELECTING JUDGES ---")
        judge_count = min(len(eligible_models), max_judges)
        logger.info(f"Randomly selecting {judge_count} judges from {len(eligible_models)} remaining eligible models")
        
        judges = random.sample(eligible_models, judge_count)
        
        logger.info(f"Selected {len(judges)} judges:")
        for i, judge in enumerate(judges, 1):
            logger.info(f"  Judge {i}: {judge.name}")
            logger.info(f"    {score_label}: {score_field(judge):.1f}")
            logger.info(f"    Previous judgments: {len(judge.match_ids['judged'])}")
        
        logger.info(f"\nRole assignment complete in initial state mode")
        logger.info(f"{'='*72}")
        
        return {
            "case_generator": case_generator,
            "question_generator": question_generator,
            "judges": judges
        }
    
    # Normal tournament state - first sort models by score
    logger.info("\n--- TOP MODELS IDENTIFICATION ---")
    logger.info(f"Calculating top models pool using {score_label}")
    all_sorted_models = sorted(models, key=sort_key, reverse=True)
    
    logger.info(f"\nAll models sorted by {score_label}:")
    for i, model in enumerate(all_sorted_models, 1):
        logger.info(f"  {i}. {model.name}: {score_label}={score_field(model):.1f}")
    
    # Calculate sizes to determine which approach to use
    quartile_size = math.ceil(len(all_sorted_models) * 0.25)  # 25% of models, rounded up
    min_top_size = 7  # Minimum top 7 models
    
    # Use whichever is higher: top 7 or top 25%
    top_size = max(min_top_size, quartile_size)
    top_size = min(top_size, len(all_sorted_models))  # Make sure we don't exceed the total number of models
    
    logger.info(f"\nDetermining selection pool size:")
    logger.info(f"  Total models: {len(all_sorted_models)}")
    logger.info(f"  25% quartile size (ceil): {quartile_size}")
    logger.info(f"  Minimum size: {min_top_size}")
    logger.info(f"  Selected size (max of the two): {top_size}")
    
    # Select top models
    global_top_models = all_sorted_models[:top_size]
    
    # Log details about the top models pool (without mentioning any ELO cutoff)
    logger.info(f"  Selected top {top_size} models for role assignment")
    for i, model in enumerate(global_top_models, 1):
        logger.info(f"    {i}. {model.name}: {score_label}={score_field(model):.1f}")
    
    # Exclude players from the top models pool
    excluded_names = {player_1.name, player_2.name}
    logger.info(f"\nExcluding match participants: {', '.join(excluded_names)}")
    
    eligible_top_models = [m for m in global_top_models if m.name not in excluded_names]
    
    logger.info(f"Found {len(eligible_top_models)}/{len(global_top_models)} eligible models in top pool after excluding players")
    logger.info(f"Eligible top models: {', '.join(m.name for m in eligible_top_models)}")
    
    # Fall back to larger pool if top models group doesn't have enough eligible models
    if len(eligible_top_models) < 2:
        logger.warning("\n*** FALLBACK REQUIRED ***")
        logger.warning("Not enough eligible models in top models pool. Using remaining models not in the match.")
        eligible_models = [m for m in models if m.name not in excluded_names]
        eligible_sorted = sorted(eligible_models, key=sort_key, reverse=True)
        eligible_top_models = eligible_sorted[:min(len(eligible_sorted), top_size)]
        logger.info(f"Selected alternative eligible models: {', '.join(m.name for m in eligible_top_models)}")
    
    if len(eligible_top_models) < 2:
        error_msg = "Not enough eligible models to assign case and question generators."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # ALWAYS RANDOMLY SELECT from top models for all roles
    logger.info("\n--- RANDOM SELECTION FROM TOP MODELS ---")
    logger.info(f"Selecting from top {len(eligible_top_models)} eligible models")
    
    # Step 1: Randomly select case generator from top models
    logger.info("\n--- SELECTING CASE GENERATOR ---")
    logger.info(f"Randomly selecting one model from {len(eligible_top_models)} eligible top models")
    
    case_generator = random.choice(eligible_top_models)
    logger.info(f"Selected case generator: {case_generator.name}")
    logger.info(f"  Rank in top models: {global_top_models.index(case_generator) + 1}/{len(global_top_models)}")
    logger.info(f"  {score_label}: {score_field(case_generator):.1f}")
    logger.info(f"  Previous cases generated: {len(case_generator.match_ids['cases_generated'])}")
    
    # Step 2: Remove case generator and randomly select question generator from remaining top models
    remaining_top = [m for m in eligible_top_models if m.name != case_generator.name]
    
    logger.info("\n--- SELECTING QUESTION GENERATOR ---")
    logger.info(f"Randomly selecting one model from {len(remaining_top)} remaining eligible top models")
    
    question_generator = random.choice(remaining_top)
    logger.info(f"Selected question generator: {question_generator.name}")
    logger.info(f"  Rank in top models: {global_top_models.index(question_generator) + 1}/{len(global_top_models)}")
    logger.info(f"  {score_label}: {score_field(question_generator):.1f}")
    logger.info(f"  Previous questions generated: {len(question_generator.match_ids['questions_generated'])}")
    
    # Step 3: Randomly select judges from remaining top models
    exclusion_set = excluded_names | {case_generator.name, question_generator.name}
    logger.info("\n--- SELECTING JUDGES ---")
    logger.info(f"Excluding all assigned roles: {', '.join(exclusion_set)}")
    
    judge_pool = [m for m in global_top_models if m.name not in exclusion_set]
    
    logger.info(f"Found {len(judge_pool)}/{len(global_top_models)} eligible judge candidates from top models")
    
    # Randomly select up to max_judges
    if len(judge_pool) > max_judges:
        logger.info(f"Randomly selecting {max_judges} judges from pool of {len(judge_pool)} eligible top models")
        judges = random.sample(judge_pool, max_judges)
    else:
        logger.info(f"Using all {len(judge_pool)} eligible top models as judges (fewer than {max_judges} max)")
        judges = judge_pool
    
    logger.info(f"\nRandomly selected {len(judges)} judges from top models:")
    for i, judge in enumerate(judges, 1):
        judge_rank = global_top_models.index(judge) + 1
        judge_counts = len(judge.match_ids["judged"])
        logger.info(f"  Judge {i}: {judge.name}")
        logger.info(f"    Rank in top models: {judge_rank}/{len(global_top_models)}")
        logger.info(f"    {score_label}: {score_field(judge):.1f}")
        logger.info(f"    Previous judgments: {judge_counts}")
    
    if len(judges) < 3:
        logger.warning(f"\nWARNING: Only {len(judges)} judges available from top models.")
        logger.warning(f"This may affect the reliability of the judgment.")
    
    logger.info(f"\nRole assignment complete")
    logger.info(f"{'='*72}")
    
    return {
        "case_generator": case_generator,
        "question_generator": question_generator,
        "judges": judges
    }

async def get_judge_vote(judge, judge_prompt, first_model, second_model):
    """Async function to get a vote from a judge."""
    logger.info(f"\n{'-'*20} JUDGE {judge.name} EVALUATION {'-'*20}")
    # Use async_generate instead of generate
    judge_result = await judge.async_generate(judge_prompt)
    vote_text = judge_result["response"].strip()
    logger.info(f"RESPONSE FROM JUDGE {judge.name}:\n{'-'*50}\n{vote_text}\n{'-'*50}")
    
    # Check for the required standardized verdict format
    import re
    verdict_pattern = r"VERDICT:\s*(Response\s*A\s*is\s*superior|Response\s*B\s*is\s*superior|This\s*is\s*a\s*tie)"
    verdict_match = re.search(verdict_pattern, vote_text, re.IGNORECASE)
    
    if verdict_match:
        # Valid verdict format found
        verdict = verdict_match.group(1).lower()
        logger.info(f"Found explicit verdict: '{verdict}'")
        
        if "response a is superior" in verdict:
            vote = first_model.model_id  # model_a is Response A
            vote_type = "win"
            logger.info(f"Judge {judge.name} explicitly voted for Response A ({first_model.name})")
        elif "response b is superior" in verdict:
            vote = second_model.model_id  # model_b is Response B
            vote_type = "win"
            logger.info(f"Judge {judge.name} explicitly voted for Response B ({second_model.name})")
        elif "this is a tie" in verdict:
            vote = "draw"
            vote_type = "draw"
            logger.info(f"Judge {judge.name} explicitly declared a tie")
        
        return {
            "judge": judge,
            "vote": vote,
            "vote_type": vote_type,
            "verdict_valid": True,
            "verdict_text": verdict,
            "prompt": judge_prompt,
            "response": vote_text
        }
    else:
        # Invalid verdict format - apply penalties
        logger.warning(f"Judge {judge.name} failed to provide a properly formatted verdict")
        logger.warning(f"No match for required pattern: {VALID_VERDICT_PATTERN}")
        
        # Apply ELO penalty to judge on BOTH rating tracks
        ELO_PENALTY = 10  # 10 points as specified in the requirements
        judge.elo["raw"]["current"] -= ELO_PENALTY
        judge.elo["cost_adjusted"]["current"] -= ELO_PENALTY  # Apply to cost-adjusted track as well
        logger.warning(f"Applied ELO penalty of {ELO_PENALTY} points to {judge.name} on both ELO tracks")
        judge.save_to_db()
        
        # Return invalid verdict result
        return {
            "judge": judge,
            "vote": None,
            "vote_type": "invalid",
            "verdict_valid": False,
            "verdict_text": None,
            "prompt": judge_prompt,
            "response": vote_text
        }

# Helper function to run async tasks within a synchronous context
def run_async_tasks(tasks):
    """Run async tasks in an event loop and return results."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(asyncio.gather(*tasks))
        return results
    finally:
        loop.close()

def run_tournament_matches(models: List[LLMModel], max_matches: int = 20, prior_matches: Optional[Set[Tuple[str, str]]] = None) -> List[Match]:
    """
    Run tournament matches until each model plays exactly 1 match.
    
    Args:
        models: List of all models
        max_matches: Maximum number of matches to run (used as safety limit)
        prior_matches: Set of already played model pairs
        
    Returns:
        List of completed matches
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING TOURNAMENT MATCHES")
    logger.info(f"{'='*80}")
    
    # Log model properties to verify initialization
    logger.info("\nMODEL PROPERTIES (Per Section 3.1 of methodology):")
    
    # Create a table header with proper borders
    header = "┌─────────────────────────┬───────────────┬──────────┬──────────┬──────────┬──────────┬────────┐"
    column_headers = f"│ {'Model Name':<23} │ {'Provider':<13} │ {'In Cost':<8} │ {'Out Cost':<8} │ {'Raw ELO':<8} │ {'Cost-ELO':<8} │ {'Matches':<6} │"
    separator = "├─────────────────────────┼───────────────┼──────────┼──────────┼──────────┼──────────┼────────┤"
    bottom = "└─────────────────────────┴───────────────┴──────────┴──────────┴──────────┴──────────┴────────┘"
    
    # Collect all model property rows
    model_rows = []
    model_info = []
    
    # Model data rows
    for model in sorted(models, key=lambda m: m.name):
        model_row = (f"│ {model.name:<23} │ {model.provider:<13} │"
                  f" ${model.input_cost_per_million:<7.6f} │ ${model.output_cost_per_million:<7.6f} │ {model.elo['raw']['current']:<8.1f} │"
                  f" {model.elo['cost_adjusted']['current']:<8.1f} │ {model.performance['total_matches_played']:<6} │")
        model_rows.append(model_row)
        
        # Build model info string
        model_info_parts = [f"Model {model.name}:"]
        model_info_parts.append(f"Provider: {model.provider},")
        model_info_parts.append(f"ELO: {model.elo['raw']['current']:.1f}")
        model_info.append(" ".join(model_info_parts))
    
    # Now log the complete table without interruptions
    logger.table("")
    logger.table("MODEL PROPERTIES:")
    logger.table(header)
    logger.table(column_headers)
    logger.table(separator)
    
    for row in model_rows:
        logger.table(row)
        
    logger.table(bottom)
    logger.table("")
    
    # Log standard info about models after table is complete
    for info in model_info:
        logger.info(info)
    
    # Initialize prior_matches if None
    if prior_matches is None:
        prior_matches = set()
    
    logger.info(f"\nBeginning tournament with {len(prior_matches)} prior matches recorded")
    
    # Log prior matches detail
    if prior_matches and len(prior_matches) > 0:
        logger.info("Prior match pairs:")
        for i, pair in enumerate(sorted(prior_matches), 1):
            logger.info(f"  {i}. {pair[0]} vs {pair[1]}")
    
    # Get top performers by ELO for special roles
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    
    # Select top 5 or fewer if not enough models
    top_count = min(5, len(sorted_models))
    top_models = sorted_models[:top_count]
    
    logger.info("\nTop performers:")
    for i, model in enumerate(top_models, 1):
        logger.info(f"{i}. {model.name}: ELO {model.elo['raw']['current']:.1f}, " + 
              f"Cost-Adj {model.elo['cost_adjusted']['current']:.1f}, " +
              f"Matches {model.performance['total_matches_played']}")
    
    # Track matches played by each model
    match_counts = {model.name: model.performance['total_matches_played'] for model in models}
    logger.info("\nCurrent match counts:")
    for name, count in match_counts.items():
        logger.info(f"  {name}: {count}/20 matches played")
    
    # Define target match count per model
    target_matches_per_model = 20
    
    # Collect the matches for this tournament
    matches = []
    match_count = 0
    safety_limit = max_matches  # Safety limit to prevent infinite loops
    
    # Continue until all models have played exactly 4 matches or we hit safety limit
    while (any(count < target_matches_per_model for count in match_counts.values()) and 
           match_count < safety_limit):
        
        logger.info("\n-----------------------------------------------------------")
        logger.info(f"MATCH SELECTION ITERATION {match_count + 1}")
        logger.info("-----------------------------------------------------------")
        
        # Find models that need more matches
        eligible_models = [m for m in models if match_counts[m.name] < target_matches_per_model]
        
        if len(eligible_models) < 2:
            logger.info(f"Not enough eligible models remain. Ending tournament.")
            break
            
        # Log eligible models
        logger.info(f"Eligible models that need more matches: {', '.join(m.name for m in eligible_models)}")
        
        # Select a pair of models
        logger.info("\nAttempting to select next pair of models...")
        model_a, model_b = select_pair(
            eligible_models,
            prior_matches=prior_matches
        )
        
        if not model_a or not model_b:
            logger.info("No valid pairs available. Trying with relaxed constraints...")
            # Try again with increased delta
            model_a, model_b = select_pair(
                eligible_models,
                initial_delta=100,
                max_delta=800,
                prior_matches=prior_matches
            )
            
            if not model_a or not model_b:
                logger.info("Still no valid pairs. Looking for any possible pairs...")
                # Get all possible pairs we haven't tried yet
                remaining_pairs = []
                for i, ma in enumerate(eligible_models):
                    for mb in eligible_models[i+1:]:
                        pair_key = tuple(sorted([ma.name, mb.name]))
                        if pair_key not in prior_matches:
                            remaining_pairs.append((ma, mb))
                
                if remaining_pairs:
                    logger.info(f"Found {len(remaining_pairs)} possible remaining pairs. Selecting randomly.")
                    model_a, model_b = random.choice(remaining_pairs)
                else:
                    logger.info("All possible pairs have been played. Ending tournament.")
                    break
        
        if not model_a or not model_b:
            logger.info("No more valid pairs available, even with relaxed constraints.")
            break
            
        # Add this pair to prior_matches
        pair_key = tuple(sorted([model_a.name, model_b.name]))
        prior_matches.add(pair_key)
        logger.info(f"Added pair to prior_matches: {pair_key[0]} vs {pair_key[1]}")
        
        match_count += 1
        logger.info(f"\nMatch {match_count}: {model_a.name} vs {model_b.name}")
        
        try:
            # Select case generator, question generator, and judges
            selections = choose_case_question_judges(
                models, model_a, model_b, max_judges=5
            )
            
            case_generator = selections["case_generator"]
            question_generator = selections["question_generator"]
            judges = selections["judges"]
            
            logger.info(f"Case generator: {case_generator.name}")
            logger.info(f"Question generator: {question_generator.name}")
            logger.info(f"Judges: {', '.join(j.name for j in judges)}")
            
            if not judges:
                logger.error("No judges available from top models for this match. Skipping.")
                continue
            
            # Generate case
            logger.info(f"\n{'='*30} CASE GENERATION {'='*30}")
            logger.info(f"Case Generator: {case_generator.name}")
            case_prompt = generate_case_prompt()
            logger.info(f"PROMPT TO CASE GENERATOR:\n{'-'*50}\n{case_prompt}\n{'-'*50}")
            case_result = case_generator.generate(case_prompt)
            case_text = case_result["response"]
            logger.info(f"RESPONSE FROM CASE GENERATOR:\n{'-'*50}\n{case_text}\n{'-'*50}")
            
            # Generate question
            logger.info(f"\n{'='*30} QUESTION GENERATION {'='*30}")
            logger.info(f"Question Generator: {question_generator.name}")
            question_prompt = generate_question_prompt(case_text)
            logger.info(f"PROMPT TO QUESTION GENERATOR:\n{'-'*50}\n{question_prompt}\n{'-'*50}")
            question_result = question_generator.generate(question_prompt)
            question_text = question_result["response"]
            logger.info(f"RESPONSE FROM QUESTION GENERATOR:\n{'-'*50}\n{question_text}\n{'-'*50}")
            
            # Create match object
            match = Match(model_a=model_a, model_b=model_b)
            
            # Set prompt
            match.set_prompt(
                case_text, 
                question_text,
                case_generator, 
                question_generator
            )
            
            # Record stratum information
            elo_avg = (model_a.elo["raw"]["current"] + model_b.elo["raw"]["current"]) / 2
            stratum_base = int(elo_avg / 100) * 100
            stratum = f"{stratum_base}-{stratum_base + 100}"
            match.set_stratum(stratum, match_count)
            logger.info(f"Match stratum: {stratum}")
            
            # Execute the full match prompt
            full_prompt = generate_answer_prompt(case_text, question_text)
            logger.info(f"\n{'='*30} PLAYER PROMPTS {'='*30}")
            logger.info(f"FULL PROMPT TO PLAYERS:\n{'-'*50}\n{full_prompt}\n{'-'*50}")
            
            # Get responses from both models
            logger.info(f"\n{'='*30} PLAYER A RESPONSE {'='*30}")
            logger.info(f"Player A: {model_a.name}")
            result_a = model_a.generate(full_prompt)
            logger.info(f"RESPONSE FROM PLAYER A ({model_a.name}):\n{'-'*50}\n{result_a['response']}\n{'-'*50}")
            
            logger.info(f"\n{'='*30} PLAYER B RESPONSE {'='*30}")
            logger.info(f"Player B: {model_b.name}")
            result_b = model_b.generate(full_prompt)
            logger.info(f"RESPONSE FROM PLAYER B ({model_b.name}):\n{'-'*50}\n{result_b['response']}\n{'-'*50}")
            
            # Record responses
            match.collect_response(
                model_a, 
                result_a["response"],
                result_a["total_tokens"],
                result_a["cost"],  # Use the cost already calculated in the model.generate method
                input_tokens=result_a["prompt_tokens"],
                output_tokens=result_a["completion_tokens"],
                input_cost=result_a["input_cost"],
                output_cost=result_a["output_cost"]
            )
            
            match.collect_response(
                model_b, 
                result_b["response"],
                result_b["total_tokens"],
                result_b["cost"],  # Use the cost already calculated in the model.generate method
                input_tokens=result_b["prompt_tokens"],
                output_tokens=result_b["completion_tokens"],
                input_cost=result_b["input_cost"],
                output_cost=result_b["output_cost"]
            )
            
            # Collect votes from judges
            judge_votes = {}
            logger.info(f"\n{'='*30} JUDGE EVALUATIONS {'='*30}")
            logger.info(f"Collecting votes from {len(judges)} judges")
            
            # Create a single judge prompt with the original responses
            # This will be used as the base template - each judge will receive their own 
            # version with potentially randomized order
            judge_prompt = generate_judge_prompt(case_text, question_text, 
                                              result_a["response"], result_b["response"])
            logger.info(f"\n{'='*30} JUDGE PROMPT {'='*30}")
            logger.info(f"JUDGE PROMPT:\n{'-'*50}\n{judge_prompt}\n{'-'*50}")
            logger.info(f"Model A is {model_a.name}, Model B is {model_b.name}")
            
            # Start timing for judge evaluations
            judge_eval_start = time.time()
            
            # Track actual model identities for easier result interpretation
            model_key_to_name = {
                model_a.model_id: model_a.name,
                model_b.model_id: model_b.name
            }
            logger.info(f"MODEL IDENTITIES: Response A is {model_a.name}, Response B is {model_b.name}")
            
            # Prepare judge tasks for async execution
            judge_tasks = []
            
            for judge in judges:
                logger.info(f"Assigning judge: {judge.name}")
                
                # Create task for this judge using the consistent prompt
                judge_tasks.append(get_judge_vote(
                    judge, 
                    judge_prompt, 
                    model_a,  # model_a is always Response A 
                    model_b   # model_b is always Response B
                ))
            
            # Run all judge evaluations concurrently
            logger.info(f"Running {len(judge_tasks)} judge evaluations concurrently")
            judge_results = run_async_tasks(judge_tasks)
            
            # End timing and calculate duration
            judge_eval_end = time.time()
            judge_eval_duration = judge_eval_end - judge_eval_start
            avg_judge_time = judge_eval_duration / len(judges) if judges else 0
            
            logger.info(f"Received {len(judge_results)} judge results")
            logger.info(f"Total judge evaluation time: {judge_eval_duration:.2f}s")
            logger.info(f"Average time per judge: {avg_judge_time:.2f}s (effective parallelization)")
            logger.info(f"Estimated sequential time: {avg_judge_time * len(judges):.2f}s")
            logger.info(f"Estimated time saved: {(avg_judge_time * len(judges)) - judge_eval_duration:.2f}s")
            
            # Process judge results 
            logger.info(f"\n{'='*30} VOTE PROCESSING {'='*30}")
            logger.info(f"Processing votes for match: {model_a.name} vs {model_b.name}")
            
            # Create summary table of votes for clarity
            vote_summary = {
                model_a.name: 0,
                model_b.name: 0,
                "tie": 0,
                "invalid": 0
            }
            
            # Track judges with invalid verdicts to remove them from consideration
            invalid_judges = []
            valid_judge_results = []
            
            # First identify invalid judgments
            for result in judge_results:
                judge = result["judge"]
                verdict_valid = result.get("verdict_valid", True)  # Default to True for backward compatibility
                
                if not verdict_valid:
                    logger.warning(f"Excluding judge {judge.name} due to invalid verdict format")
                    invalid_judges.append(judge)
                    vote_summary["invalid"] += 1
                else:
                    valid_judge_results.append(result)
            
            # If all judges provided invalid verdicts, we can't proceed
            if len(valid_judge_results) == 0:
                logger.error("No valid judge verdicts found! Cannot determine match outcome.")
                logger.error("Assigning random outcome as fallback.")
                random_outcome = random.choice(["a", "b", "draw"])
                if random_outcome == "a":
                    judge_votes = {model_a.model_id: 1, model_b.model_id: 0}
                elif random_outcome == "b":
                    judge_votes = {model_a.model_id: 0, model_b.model_id: 1}
                else:
                    judge_votes = {model_a.model_id: 0.5, model_b.model_id: 0.5}
            else:
                # Process valid judge results only
                judge_votes = {}
                
                for result in valid_judge_results:
                    judge = result["judge"]
                    vote = result["vote"]
                    vote_type = result["vote_type"]
                    judge_prompt = result.get("prompt", "")
                    judge_response = result.get("response", "")
                    
                    # Determine which actual model was voted for
                    if vote_type == "win":
                        voted_model_name = model_key_to_name.get(vote, "unknown")
                        logger.info(f"Judge {judge.name} voted for {voted_model_name}")
                        vote_summary[voted_model_name] += 1
                        
                        # Add to judge votes
                        if vote not in judge_votes:
                            judge_votes[vote] = 0
                        judge_votes[vote] += 1
                    elif vote_type == "draw":
                        logger.info(f"Judge {judge.name} declared a draw/tie")
                        vote_summary["tie"] += 1
                        
                        # Add half a vote to each model for draws
                        if model_a.model_id not in judge_votes:
                            judge_votes[model_a.model_id] = 0
                        if model_b.model_id not in judge_votes:
                            judge_votes[model_b.model_id] = 0
                        judge_votes[model_a.model_id] += 0.5
                        judge_votes[model_b.model_id] += 0.5
                    
                    # Record vote in match - only for valid votes
                    weight = 1.0 / len(valid_judge_results)  # Recalculate weights based on valid judges only
                    match.add_judge(judge, vote, weight, vote_type, 
                                   prompt=judge_prompt, 
                                   response=judge_response)
                
                # Log the vote summary table
                logger.info("\nVOTE SUMMARY:")
                for category, count in vote_summary.items():
                    logger.info(f"  {category}: {count} votes")
                
                if invalid_judges:
                    logger.warning(f"\n{len(invalid_judges)} judges provided invalid verdicts and were excluded:")
                    for j in invalid_judges:
                        logger.warning(f"  - {j.name}")
                
                # Calculate final scores based on judge votes
                model_a_id = model_a.model_id
                model_b_id = model_b.model_id
                
                # Ensure both models have entries in judge_votes
                if model_a_id not in judge_votes:
                    judge_votes[model_a_id] = 0
                if model_b_id not in judge_votes:
                    judge_votes[model_b_id] = 0
                
                # Raw score is proportion of votes (with draws counting as 0.5)
                total_votes = judge_votes[model_a_id] + judge_votes[model_b_id]
                if total_votes == 0:
                    logger.error("No valid votes cast. Defaulting to equal scores.")
                    raw_scores = {model_a_id: 0.5, model_b_id: 0.5}
                else:
                    raw_scores = {
                        model_a_id: judge_votes[model_a_id] / total_votes,
                        model_b_id: judge_votes[model_b_id] / total_votes
                    }
                
                logger.info(f"Raw scores from unweighted votes: {model_a.name}: {raw_scores.get(model_a_id, 0):.2f}, {model_b.name}: {raw_scores.get(model_b_id, 0):.2f}")
                
                # ===== VOTE WEIGHTING AND SCORE CALCULATION =====
                logger.info(f"\n{'='*30} JUDGMENT AND SCORING {'='*30}")
                logger.info(f"Following Section 3.6 (Federated Judgment by Peer Models)")
                logger.info(f"Using {len(judges)} judges with raw-ELO softmax weighting")
                logger.info(f"Match: {model_a.name} vs {model_b.name}")

            # Calculate judge weights using softmax (Equation 8)
            judge_elos = [j.elo["raw"]["current"] for j in judges]
            temperature = 300  # softmax temperature from paper τ
            
            # Show the numerator calculations
            logger.info("\nJUDGE WEIGHT CALCULATIONS (using raw ELO)")
            logger.info(f"w_k = e^(R_k^raw/τ) / Σ_j=1^J e^(R_j^raw/τ) with τ = {temperature}")
            logger.info("\nNumerator calculations for each judge (e^(R_k^raw/τ)):")
            numerators = []
            for i, (judge, elo) in enumerate(zip(judges, judge_elos)):
                numerator = math.exp(elo / temperature)
                numerators.append(numerator)
                logger.info(f"  {judge.name}: e^({elo:.1f}/{temperature}) = {numerator:.6f}")
            
            # Calculate the denominator (sum of all numerators)
            denominator = sum(numerators)
            logger.info(f"\nDenominator calculation (sum of all numerators): {denominator:.6f}")
            
            # Calculate the final weights (numerator / denominator)
            normalized_weights = []
            logger.info("\nFinal normalized weights:")
            for i, (judge, num) in enumerate(zip(judges, numerators)):
                weight = num / denominator
                normalized_weights.append(weight)
                logger.info(f"  {judge.name}: {num:.6f}/{denominator:.6f} = {weight:.6f} ({weight*100:.2f}%)")
            
            # ===== TABULATE ALL VOTES AND CALCULATE RAW SCORE =====
            logger.info(f"\nEQUATION 9-10: CALCULATING WEIGHTED SCORE")
            logger.info(f"S_raw^A = Σ_k=1^J w_k * v_k^A  (where v_k^A is 1 if judge k voted for A, 0 otherwise)")
            
            # Count votes for each model
            vote_a_count = 0
            vote_b_count = 0
            draw_count = 0
            
            # Initialize weighted scores
            raw_score_a = 0.0
            raw_score_b = 0.0
            
            # Judge votes table
            logger.info("\nJudge votes and weighted contributions:")
            
            # Table header with borders
            vote_header = "┌───────────────┬───────────────┬──────────┬──────────┬───────────────┬───────────────┐"
            vote_column_headers = f"│ {'Judge':<13} │ {'Vote For':<13} │ {'Vote Type':<8} │ {'Weight':<8} │ {'Contribution A':<13} │ {'Contribution B':<13} │"
            vote_separator = "├───────────────┼───────────────┼──────────┼──────────┼───────────────┼───────────────┤"
            vote_bottom = "└───────────────┴───────────────┴──────────┴──────────┴───────────────┴───────────────┘"
            
            # Collect all vote rows first
            vote_rows = []
            vote_info = []
            
            # Process each judge's vote and calculate weighted contribution
            for i, (judge, weight) in enumerate(zip(judges, normalized_weights)):
                vote = match.judgment["judges"][i]["vote"]
                vote_type = match.judgment["judges"][i]["vote_type"]
                
                if vote_type == "win":
                    if vote == model_a.model_id:
                        vote_a_count += 1
                        vote_for = model_a.name
                        contrib_a = weight
                        contrib_b = 0
                        raw_score_a += weight
                    else:
                        vote_b_count += 1
                        vote_for = model_b.name
                        contrib_a = 0
                        contrib_b = weight
                        raw_score_b += weight
                else:  # draw
                    draw_count += 1
                    vote_for = "DRAW"
                    # For draws, split the weight equally
                    contrib_a = weight / 2
                    contrib_b = weight / 2
                    raw_score_a += contrib_a
                    raw_score_b += contrib_b
                
                # Add to vote rows for table
                vote_row = f"│ {judge.name:<13} │ {vote_for:<13} │ {vote_type:<8} │ {weight:<8.6f} │ {contrib_a:<13.6f} │ {contrib_b:<13.6f} │"
                vote_rows.append(vote_row)
                
                # Keep existing judge entry but update the weight
                # (original prompt and response will be preserved from the first add_judge call)
                judge_entry = next((j for j in match.judgment["judges"] if j["judge_id"] == judge.model_id), None)
                if judge_entry:
                    judge_entry["weight"] = weight
                                
                # Add info for standard logging
                vote_info.append(f"Judge {judge.name} voted for {vote_for} with weight {weight:.6f}, contributing {contrib_a:.6f} to A and {contrib_b:.6f} to B")
            
            # Now log the complete table without interruptions
            logger.table("")
            logger.table("JUDGE VOTES AND CONTRIBUTIONS:")
            logger.table(vote_header)
            logger.table(vote_column_headers)
            logger.table(vote_separator)
            
            for row in vote_rows:
                logger.table(row)
                
            logger.table(vote_bottom)
            logger.table("")
            
            # Log standard info about votes after table is complete
            for info in vote_info:
                logger.info(info)
            
            logger.info(f"\nVote distribution:")
            logger.info(f"  {model_a.name}: {vote_a_count}/{len(judges)} votes")
            logger.info(f"  {model_b.name}: {vote_b_count}/{len(judges)} votes")
            logger.info(f"  Draws: {draw_count}/{len(judges)} votes")
            
            logger.info(f"\nFinal raw scores (Equation 9):")
            logger.info(f"  S_raw^{model_a.name} = {raw_score_a:.6f}")
            logger.info(f"  S_raw^{model_b.name} = {raw_score_b:.6f}")
            
            # Record raw scores
            match.judgment["raw_score_a"] = raw_score_a
            match.judgment["raw_score_b"] = raw_score_b
            
            # ===== CALCULATE COST-ADJUSTED SCORES (EQUATION 11) =====
            logger.info(f"\nEQUATION 11: COST-ADJUSTED SCORES")
            logger.info(f"S_adj^A = S_raw^A * (c_avg/c_A)^0.5")
            
            # Calculate response costs
            cost_a = result_a["cost"]
            cost_b = result_b["cost"]
            
            logger.info(f"\nResponse costs:")
            logger.info(f"  {model_a.name}: ${cost_a:.6f}")
            logger.info(f"  {model_b.name}: ${cost_b:.6f}")
            
            # Calculate efficiency weights following equation 16
            temperature_c = 0.05  # τ_c value from the paper
            
            # Efficiency weight of A equals e to the minus cost of A over τ_c, 
            # divided by [e to the minus cost of A over τ_c plus e to the minus cost of B over τ_c]
            numerator_a = math.exp(-cost_a / temperature_c)  # e^(-cost_A/τ_c)
            numerator_b = math.exp(-cost_b / temperature_c)  # e^(-cost_B/τ_c)
            
            denominator = numerator_a + numerator_b  # Sum of the two exponentials
            
            # Complete efficiency weights calculation
            eff_a = numerator_a / denominator
            eff_b = numerator_b / denominator
            
            logger.info(f"\nEfficiency weights calculation (equation 16):")
            logger.info(f"  numerator_A = e^(-cost_A/τ_c) = e^(-{cost_a:.6f}/{temperature_c}) = {numerator_a:.6f}")
            logger.info(f"  numerator_B = e^(-cost_B/τ_c) = e^(-{cost_b:.6f}/{temperature_c}) = {numerator_b:.6f}")
            logger.info(f"  denominator = numerator_A + numerator_B = {numerator_a:.6f} + {numerator_b:.6f} = {denominator:.6f}")
            logger.info(f"  eff_A = numerator_A / denominator = {numerator_a:.6f} / {denominator:.6f} = {eff_a:.6f}")
            logger.info(f"  eff_B = numerator_B / denominator = {numerator_b:.6f} / {denominator:.6f} = {eff_b:.6f}")
            
            # Cost-adjusted match score of A equals (S_raw_A times eff_A) divided by
            # [S_raw_A times eff_A plus S_raw_B times eff_B]
            
            # Calculate terms for cost-adjusted scores using equation 17
            term_a = raw_score_a * eff_a  # S_raw_A * eff_A
            term_b = raw_score_b * eff_b  # S_raw_B * eff_B
            
            # Sum of the terms for the denominator
            score_denominator = term_a + term_b
            
            # Calculate the final adjusted scores
            adj_score_a = term_a / score_denominator if score_denominator > 0 else 0.5
            adj_score_b = term_b / score_denominator if score_denominator > 0 else 0.5
            
            logger.info(f"\nCost-adjusted scores calculation (Equation 17):")
            logger.info(f"  term_A = S_raw_A * eff_A = {raw_score_a:.6f} * {eff_a:.6f} = {term_a:.6f}")
            logger.info(f"  term_B = S_raw_B * eff_B = {raw_score_b:.6f} * {eff_b:.6f} = {term_b:.6f}")
            logger.info(f"  denominator = term_A + term_B = {term_a:.6f} + {term_b:.6f} = {score_denominator:.6f}")
            logger.info(f"  S_adj_A = term_A / denominator = {term_a:.6f} / {score_denominator:.6f} = {adj_score_a:.6f}")
            logger.info(f"  S_adj_B = term_B / denominator = {term_b:.6f} / {score_denominator:.6f} = {adj_score_b:.6f}")
            
            # Record adjusted scores in the correct format for the match object
            raw_scores = {
                model_a_id: raw_score_a,
                model_b_id: raw_score_b
            }
            cost_adjusted_scores = {
                model_a_id: adj_score_a,
                model_b_id: adj_score_b
            }
            
            # Set the scores in the match object
            match.set_scores(raw_scores, cost_adjusted_scores)
            
            # ===== UPDATE ELO RATINGS =====
            logger.info(f"\n{'='*30} ELO RATING UPDATE {'='*30}")
            logger.info(f"Following exact step-by-step calculation process")
            
            # K-factor from paper
            K = 16
            logger.info(f"\nUsing K-factor = {K} as specified")
            
            # ===== ELO RATING VARIABLES =====
            # Current ratings
            old_raw_rating_of_A = model_a.elo["raw"]["current"]
            old_raw_rating_of_B = model_b.elo["raw"]["current"]
            old_cost_rating_of_A = model_a.elo["cost_adjusted"]["current"]
            old_cost_rating_of_B = model_b.elo["cost_adjusted"]["current"]
            
            logger.info(f"\nCurrent ELO ratings:")
            logger.info(f"  Old raw rating of A ({model_a.name}): {old_raw_rating_of_A:.1f}")
            logger.info(f"  Old raw rating of B ({model_b.name}): {old_raw_rating_of_B:.1f}")
            logger.info(f"  Old cost rating of A ({model_a.name}): {old_cost_rating_of_A:.1f}")
            logger.info(f"  Old cost rating of B ({model_b.name}): {old_cost_rating_of_B:.1f}")
            
            # ===== EXPECTED SCORES =====
            # Expected raw score of A = 1 divided by [1 plus 10 to the power of (raw rating of B minus raw rating of A divided by 400)]
            expected_raw_score_of_A = 1.0 / (1.0 + math.pow(10, (old_raw_rating_of_B - old_raw_rating_of_A) / 400.0))
            
            # Expected cost score of A = 1 divided by [1 plus 10 to the power of (cost rating of B minus cost rating of A divided by 400)]
            expected_cost_score_of_A = 1.0 / (1.0 + math.pow(10, (old_cost_rating_of_B - old_cost_rating_of_A) / 400.0))
            
            # Expected raw score of B = 1 minus expected raw score of A
            expected_raw_score_of_B = 1.0 - expected_raw_score_of_A
            
            # Expected cost score of B = 1 minus expected cost score of A
            expected_cost_score_of_B = 1.0 - expected_cost_score_of_A
            
            logger.info(f"\nExpected scores:")
            logger.info(f"  Expected raw score of A: {expected_raw_score_of_A:.4f}")
            logger.info(f"  Expected raw score of B: {expected_raw_score_of_B:.4f}")
            logger.info(f"  Expected cost score of A: {expected_cost_score_of_A:.4f}")
            logger.info(f"  Expected cost score of B: {expected_cost_score_of_B:.4f}")
            
            # ===== ACTUAL SCORES =====
            # Actual scores from the match
            actual_raw_score_of_A = raw_score_a
            
            # Actual raw score of B = 1 minus actual raw score of A
            actual_raw_score_of_B = 1.0 - actual_raw_score_of_A
            
            # Actual cost-adjusted scores 
            actual_cost_adjusted_score_of_A = adj_score_a
            
            # Actual cost-adjusted score of B = 1 minus actual cost-adjusted score of A
            actual_cost_adjusted_score_of_B = 1.0 - actual_cost_adjusted_score_of_A
            
            logger.info(f"\nActual scores from the match:")
            logger.info(f"  Actual raw score of A: {actual_raw_score_of_A:.4f}")
            logger.info(f"  Actual raw score of B: {actual_raw_score_of_B:.4f}")
            logger.info(f"  Actual cost-adjusted score of A: {actual_cost_adjusted_score_of_A:.4f}")
            logger.info(f"  Actual cost-adjusted score of B: {actual_cost_adjusted_score_of_B:.4f}")
            
            # ===== RATING UPDATES =====
            # New raw rating of A = old raw rating of A plus K times (actual raw score of A minus expected raw score of A)
            new_raw_rating_of_A = old_raw_rating_of_A + K * (actual_raw_score_of_A - expected_raw_score_of_A)
            
            # New cost rating of A = old cost rating of A plus K times (actual cost-adjusted score of A minus expected cost score of A)
            new_cost_rating_of_A = old_cost_rating_of_A + K * (actual_cost_adjusted_score_of_A - expected_cost_score_of_A)
            
            # New raw rating of B = old raw rating of B plus K times (actual raw score of B minus expected raw score of B)
            new_raw_rating_of_B = old_raw_rating_of_B + K * (actual_raw_score_of_B - expected_raw_score_of_B)
            
            # New cost rating of B = old cost rating of B plus K times (actual cost-adjusted score of B minus expected cost score of B)
            new_cost_rating_of_B = old_cost_rating_of_B + K * (actual_cost_adjusted_score_of_B - expected_cost_score_of_B)
            
            # Calculate changes for logging
            raw_rating_change_of_A = new_raw_rating_of_A - old_raw_rating_of_A
            raw_rating_change_of_B = new_raw_rating_of_B - old_raw_rating_of_B
            cost_rating_change_of_A = new_cost_rating_of_A - old_cost_rating_of_A
            cost_rating_change_of_B = new_cost_rating_of_B - old_cost_rating_of_B
            
            logger.info(f"\nRating update calculations:")
            logger.info(f"  Raw rating change of A: {K} × ({actual_raw_score_of_A:.4f} - {expected_raw_score_of_A:.4f}) = {raw_rating_change_of_A:.1f}")
            logger.info(f"  Raw rating change of B: {K} × ({actual_raw_score_of_B:.4f} - {expected_raw_score_of_B:.4f}) = {raw_rating_change_of_B:.1f}")
            logger.info(f"  Cost rating change of A: {K} × ({actual_cost_adjusted_score_of_A:.4f} - {expected_cost_score_of_A:.4f}) = {cost_rating_change_of_A:.1f}")
            logger.info(f"  Cost rating change of B: {K} × ({actual_cost_adjusted_score_of_B:.4f} - {expected_cost_score_of_B:.4f}) = {cost_rating_change_of_B:.1f}")
            
            logger.info(f"\nNew ELO ratings:")
            logger.info(f"  New raw rating of A ({model_a.name}): {old_raw_rating_of_A:.1f} + {raw_rating_change_of_A:.1f} = {new_raw_rating_of_A:.1f}")
            logger.info(f"  New raw rating of B ({model_b.name}): {old_raw_rating_of_B:.1f} + {raw_rating_change_of_B:.1f} = {new_raw_rating_of_B:.1f}")
            logger.info(f"  New cost rating of A ({model_a.name}): {old_cost_rating_of_A:.1f} + {cost_rating_change_of_A:.1f} = {new_cost_rating_of_A:.1f}")
            logger.info(f"  New cost rating of B ({model_b.name}): {old_cost_rating_of_B:.1f} + {cost_rating_change_of_B:.1f} = {new_cost_rating_of_B:.1f}")
            
            # Calculate standard error after this match
            match_count_a = model_a.performance['total_matches_played'] + 1
            match_count_b = model_b.performance['total_matches_played'] + 1
            
            # Standard error after n matches σ ≈ 400 divided by √n
            error_a = 400.0 / math.sqrt(match_count_a)
            error_b = 400.0 / math.sqrt(match_count_b)
            
            logger.info(f"\nStatistical confidence:")
            logger.info(f"  {model_a.name}: After {match_count_a} matches, standard error ≈ {error_a:.1f} points")
            logger.info(f"  {model_b.name}: After {match_count_b} matches, standard error ≈ {error_b:.1f} points")
            
            # Save the new ratings for use later in the code
            # (Backward compatibility with existing variables)
            new_raw_elo_a = new_raw_rating_of_A
            new_raw_elo_b = new_raw_rating_of_B
            new_cost_elo_a = new_cost_rating_of_A
            new_cost_elo_b = new_cost_rating_of_B
            
            # ===== SUMMARY OF RESULTS =====
            logger.info(f"\n{'='*30} MATCH RESULTS SUMMARY {'='*30}")
            logger.info(f"Match: {model_a.name} vs {model_b.name}")
            
            # Create a match results table with clean formatting
            results_header = "┌─────────────────────┬────────────┬────────────┬────────────┬──────────┬──────────┬──────────┐"
            column_header = f"│ {'Model':<19} │ {'Raw Score':<10} │ {'Cost Score':<10} │ {'Tokens':<10} │ {'Cost $':<8} │ {'Raw ELO':<8} │ {'Cost ELO':<8} │"
            separator = "├─────────────────────┼────────────┼────────────┼────────────┼──────────┼──────────┼──────────┤"
            bottom_line = "└─────────────────────┴────────────┴────────────┴────────────┴──────────┴──────────┴──────────┘"
            
            # Log to table without timestamps
            logger.table("")
            logger.table(f"MATCH RESULTS: {model_a.name} vs {model_b.name}")
            logger.table(results_header)
            logger.table(column_header)
            logger.table(separator)
            
            # Model A stats
            total_tokens_a = result_a["prompt_tokens"] + result_a["completion_tokens"]
            logger.table(f"│ {model_a.name:<19} │ {raw_score_a:<10.4f} │ {adj_score_a:<10.4f} │ {total_tokens_a:<10} │ ${cost_a:<7.4f} │ {new_raw_elo_a:<8.1f} │ {new_cost_elo_a:<8.1f} │")
            
            # Model B stats  
            total_tokens_b = result_b["prompt_tokens"] + result_b["completion_tokens"]
            logger.table(f"│ {model_b.name:<19} │ {raw_score_b:<10.4f} │ {adj_score_b:<10.4f} │ {total_tokens_b:<10} │ ${cost_b:<7.4f} │ {new_raw_elo_b:<8.1f} │ {new_cost_elo_b:<8.1f} │")
            
            logger.table(bottom_line)
            
            # Add ELO change info
            elo_header = "┌─────────────────────┬────────────┬────────────┬────────────┬────────────┐"
            elo_column = f"│ {'Model':<19} │ {'Raw ELO Δ':<10} │ {'New Raw':<10} │ {'Cost ELO Δ':<10} │ {'New Cost':<10} │"
            elo_separator = "├─────────────────────┼────────────┼────────────┼────────────┼────────────┤"
            elo_bottom = "└─────────────────────┴────────────┴────────────┴────────────┴────────────┘"
            
            logger.table("")
            logger.table("ELO CHANGES:")
            logger.table(elo_header)
            logger.table(elo_column)
            logger.table(elo_separator)
            logger.table(f"│ {model_a.name:<19} │ {raw_rating_change_of_A:+10.1f} │ {new_raw_elo_a:<10.1f} │ {cost_rating_change_of_A:+10.1f} │ {new_cost_elo_a:<10.1f} │")
            logger.table(f"│ {model_b.name:<19} │ {raw_rating_change_of_B:+10.1f} │ {new_raw_elo_b:<10.1f} │ {cost_rating_change_of_B:+10.1f} │ {new_cost_elo_b:<10.1f} │")
            logger.table(elo_bottom)
            logger.table("")

                # Log standard results with timestamps
            logger.info(f"\nRaw scores: {model_a.name} {raw_score_a:.4f} - {raw_score_b:.4f} {model_b.name}")
            logger.info(f"Cost-adj scores: {model_a.name} {adj_score_a:.4f} - {adj_score_b:.4f} {model_b.name}")
            
            # Determine winner based on adjusted scores
            if adj_score_a > adj_score_b:
                winner = model_a.name
                win_margin = adj_score_a - adj_score_b
            elif adj_score_b > adj_score_a:
                winner = model_b.name
                win_margin = adj_score_b - adj_score_a
            else:
                winner = "DRAW"
                win_margin = 0
            
            if winner != "DRAW":
                logger.info(f"\nWINNER: {winner} by {win_margin:.4f} points")
            else:
                logger.info(f"\nRESULT: DRAW")
            
            logger.info(f"\nELO changes:")
            logger.info(f"  {model_a.name}: Raw: {old_raw_rating_of_A:.1f} → {new_raw_elo_a:.1f} ({raw_rating_change_of_A:+.1f}), " +
                        f"Cost-Adj: {old_cost_rating_of_A:.1f} → {new_cost_elo_a:.1f} ({cost_rating_change_of_A:+.1f})")
            logger.info(f"  {model_b.name}: Raw: {old_raw_rating_of_B:.1f} → {new_raw_elo_b:.1f} ({raw_rating_change_of_B:+.1f}), " +
                        f"Cost-Adj: {old_cost_rating_of_B:.1f} → {new_cost_elo_b:.1f} ({cost_rating_change_of_B:+.1f})")
            logger.info(f"{'='*72}\n")
            
            # Set the new ELO ratings to the model objects
            model_a.elo["raw"]["current"] = new_raw_elo_a
            model_a.elo["cost_adjusted"]["current"] = new_cost_elo_a
            model_b.elo["raw"]["current"] = new_raw_elo_b
            model_b.elo["cost_adjusted"]["current"] = new_cost_elo_b
            
            # Update performance metrics
            match_result_a = "win" if raw_score_a > 0.5 else "loss" if raw_score_a < 0.5 else "draw"
            match_result_b = "win" if raw_score_b > 0.5 else "loss" if raw_score_b < 0.5 else "draw"
            
            # Update model A stats
            model_a.performance["total_matches_played"] += 1
            model_a.match_ids["played"].append(match.match_id)
            if match_result_a == "win":
                model_a.performance["wins_raw"] += 1
            elif match_result_a == "loss":
                model_a.performance["losses_raw"] += 1
                
            # Update model B stats
            model_b.performance["total_matches_played"] += 1
            model_b.match_ids["played"].append(match.match_id)
            if match_result_b == "win":
                model_b.performance["wins_raw"] += 1
            elif match_result_b == "loss":
                model_b.performance["losses_raw"] += 1
                
            # Save updated models to database
            model_a.save_to_db()
            model_b.save_to_db()
            
            # Record updated ELO ratings in the match object
            match.update_elo(model_a, model_b)
            
            # Save match to database
            match.save_to_db()
            
            # Update match counts after successful match
            match_counts[model_a.name] += 1
            match_counts[model_b.name] += 1
            
            # Add to matches list
            matches.append(match)
            
            logger.info(f"Match complete: {model_a.name} vs {model_b.name}")
            logger.info(f"Updated ELO: {model_a.name}: {model_a.elo['raw']['current']:.1f}, {model_b.name}: {model_b.elo['raw']['current']:.1f}")
            
            # Log updated match counts
            logger.info("\nUpdated match counts:")
            for name, count in sorted(match_counts.items()):
                logger.info(f"  {name}: {count}/20 matches played")
                
            # Log progress toward target
            models_at_target = sum(1 for count in match_counts.values() if count >= target_matches_per_model)
            logger.info(f"\nProgress: {models_at_target}/{len(models)} models have played {target_matches_per_model} matches")
            
        except Exception as e:
            logger.error(f"ERROR in match {match_count}: {str(e)}")
            logger.info("Skipping to next match.")
            continue
    
    # Final match count report
    logger.info(f"\n{'='*30} FINAL MATCH COUNT REPORT {'='*30}")
    logger.info(f"Target matches per model: {target_matches_per_model}")
    
    # Create a match count table
    count_header = "┌─────────────────────┬──────────┐"
    count_column_headers = f"│ {'Model':<19} │ {'Matches':<8} │"
    count_separator = "├─────────────────────┼──────────┤"
    count_bottom = "└─────────────────────┴──────────┘"
    
    logger.table("")
    logger.table("FINAL MATCH COUNTS:")
    logger.table(count_header)
    logger.table(count_column_headers)
    logger.table(count_separator)
    
    for name, count in sorted(match_counts.items()):
        status = "✓" if count >= target_matches_per_model else "✗"
        logger.table(f"│ {name:<19} │ {count:>4}/{target_matches_per_model}  {status} │")
        
    logger.table(count_bottom)
    logger.table("")
    
    # Calculate total needed and completed matches
    total_needed = len(models) * target_matches_per_model
    total_completed = sum(match_counts.values())
    theoretical_max = (len(models) * target_matches_per_model) // 2
    
    logger.info(f"Completed {match_count} matches out of theoretical maximum of {theoretical_max}")
    logger.info(f"Models played {total_completed} total matches out of {total_needed} needed for complete coverage")
    
    # Check if all models reached target
    all_at_target = all(count >= target_matches_per_model for count in match_counts.values())
    if all_at_target:
        logger.info(f"SUCCESS: All models have played at least {target_matches_per_model} matches")
    else:
        models_below_target = [name for name, count in match_counts.items() if count < target_matches_per_model]
        logger.info(f"INCOMPLETE: {len(models_below_target)} models did not reach {target_matches_per_model} matches")
        logger.info(f"Models below target: {', '.join(models_below_target)}")
    
    logger.info(f"Tournament complete: {len(matches)} matches executed")
    
    return matches


# Example usage
if __name__ == "__main__":
    from models import initialize_models
    from model_definitions import MODELS
    
    # Initialize models
    all_models = initialize_models(MODELS)
    
    # Run tournament matches
    max_matches = 20
    matches = run_tournament_matches(all_models, max_matches)
    
    # Show results
    print("\nTournament Results:")
    for match in matches:
        print(match)