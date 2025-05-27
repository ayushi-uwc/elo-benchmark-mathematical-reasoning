import math
from typing import List
from models import LLMModel
from logger_config import get_logger

logger = get_logger(__name__)

def print_detailed_leaderboard(models: List[LLMModel]):
    """Print detailed leaderboard with all statistics."""
    # Sort models by raw ELO rating
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    
    # Create table header
    header = "┌───┬──────────────────┬────────┬────────┬──────────┬──────────┬────────┬────────┬───────────┬────────┐"
    column_headers = f"│{' #':<3}│{'Model':<18}│{'Raw ELO':^8}│{'Cost ELO':^8}│{'Raw Avg':^10}│{'Cost Avg':^10}│{'W-L-D':^8}│{'Tokens':^8}│{'Cost $':^9}│{'Matches':^8}│"
    separator = "├───┼──────────────────┼────────┼────────┼──────────┼──────────┼────────┼────────┼───────────┼────────┤"
    bottom = "└───┴──────────────────┴────────┴────────┴──────────┴──────────┴────────┴────────┴───────────┴────────┘"
    
    print("\nDETAILED LEADERBOARD")
    print(header)
    print(column_headers)
    print(separator)
    
    for i, model in enumerate(sorted_models, 1):
        # Calculate W-L-D string from individual judge votes
        w = model.performance["wins_raw"]
        l = model.performance["losses_raw"]
        d = model.performance["draws_raw"]
        wld = f"{w}-{l}-{d}"
        
        # Get total tokens and cost
        total_tokens = model.performance["total_tokens_used"]
        total_cost = model.performance["total_cost_usd"]
        
        matches_played = model.performance["total_matches_played"]
        
        # Calculate true ELO averages from match history
        raw_elo_avg = model.performance["score_history"]["avg_raw_score"]
        cost_elo_avg = model.performance["score_history"]["avg_adjusted_score"]
        
        print(f"│{i:^3}│{model.name:<18}│{model.elo['raw']['current']:^8.1f}│"
              f"{model.elo['cost_adjusted']['current']:^8.1f}│{raw_elo_avg:^10.4f}│{cost_elo_avg:^10.4f}│"
              f"{wld:^8}│{total_tokens:^8}│${total_cost:^8.5f}│{matches_played:^8d}│")

    print(bottom)
    print(f"\nW-L-D shows cumulative judge votes (wins-losses-draws)")
    print(f"Raw/Cost Avg show average ELO ratings across all matches")
    
    # Log the table
    logger.table("")
    logger.table("DETAILED LEADERBOARD")
    logger.table(header)
    logger.table(column_headers)
    logger.table(separator)
    
    for i, model in enumerate(sorted_models, 1):
        w = model.performance["wins_raw"]
        l = model.performance["losses_raw"]
        d = model.performance["draws_raw"]
        wld = f"{w}-{l}-{d}"
        matches_played = model.performance["total_matches_played"]
        total_tokens = model.performance["total_tokens_used"]
        total_cost = model.performance["total_cost_usd"]
        
        # Calculate true ELO averages from match history
        raw_elo_avg = model.performance["score_history"]["avg_raw_score"]
        cost_elo_avg = model.performance["score_history"]["avg_adjusted_score"]
        
        logger.table(f"│{i:^3}│{model.name:<18}│{model.elo['raw']['current']:^8.1f}│"
                    f"{model.elo['cost_adjusted']['current']:^8.1f}│{raw_elo_avg:^10.4f}│{cost_elo_avg:^10.4f}│"
                    f"{wld:^8}│{total_tokens:^8}│${total_cost:^8.5f}│{matches_played:^8d}│")
        
    logger.table(bottom)
    logger.table("")
    
    # Log individual model info
    for i, model in enumerate(sorted_models, 1):
        w = model.performance["wins_raw"]
        l = model.performance["losses_raw"]
        d = model.performance["draws_raw"]
        logger.info(f"Rank {i}: {model.name} - Raw: {model.elo['raw']['current']:.1f}, "
                   f"Cost-Adj: {model.elo['cost_adjusted']['current']:.1f}, "
                   f"Record: {w}-{l}-{d}") 