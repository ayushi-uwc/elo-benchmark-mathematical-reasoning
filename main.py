#!/usr/bin/env python3
# main.py
"""
LLM Tournament System - Main Entry Point
"""
import argparse
import sys
import time
import logging  # Add this import
from datetime import datetime
from typing import List, Dict, Any, Optional
import math

# Import logger configuration first
from logger_config import get_logger, setup_logger

# Import our modules
from database import db, check_connection_status
from models import LLMModel, initialize_models
from model_definitions import MODELS
from tournament import run_tournament_matches, get_top_performers
from leaderboard import print_detailed_leaderboard

# Get logger for this module
logger = get_logger(__name__)

# Add near the top of main.py, after logger is set up
logger.info("========== STARTING NEW TOURNAMENT RUN ==========")
# Force flush the log to ensure it's written
try:
    logger.force_flush()
except:
    pass

def print_header():
    """Print program header."""
    print("\nELO BENCHMARK SYSTEM")
    print("===================")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def print_model_stats(models: List[LLMModel], detailed: bool = False):
    """Print statistics about all models."""
    logger.info("Printing model statistics")
    
    # Sort by raw ELO score
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    
    stats_header = "\nMODEL RANKINGS:"
    
    # Create standardized table with box-drawing characters
    header_line = "┌───┬────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐"
    column_headers = f"│{'#':^3}│{'Model':^18}│{'Raw ELO':^8}│{'Cost ELO':^8}│{'Raw Avg':^8}│{'Cost Avg':^8}│{'W-L-D':^8}│{'Tot Tok':^8}│{'Tot Cost':^8}│{'Matches':^8}│{'Rating±':^8}│"
    separator = "├───┼────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤"
    bottom = "└───┴────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘"
    
    print(stats_header)
    print(header_line)
    print(column_headers)
    print(separator)
    
    for i, model in enumerate(sorted_models, 1):
        # Calculate W-L-D string
        w = model.performance["wins_raw"]
        l = model.performance["losses_raw"]
        d = model.performance["draws_raw"]
        wld = f"{w}-{l}-{d}"
        
        # Get total tokens and cost
        total_tokens = model.performance["total_tokens_used"]
        total_cost = model.performance["total_cost_usd"]
        
        # Calculate rating uncertainty (standard error)
        matches_played = model.performance["total_matches_played"]
        rating_uncertainty = 400 / math.sqrt(matches_played) if matches_played > 0 else 400
        
        # Get average scores
        avg_raw = model.performance["score_history"]["avg_raw_score"]
        avg_cost = model.performance["score_history"]["avg_adjusted_score"]
        
        print(f"│{i:^3}│{model.name:<18}│{model.elo['raw']['current']:^8.1f}│"
              f"{model.elo['cost_adjusted']['current']:^8.1f}│{avg_raw:^8.3f}│{avg_cost:^8.3f}│"
              f"{wld:^8}│{total_tokens:^8}│${total_cost:^7.2f}│{matches_played:^8d}│±{rating_uncertainty:^7.1f}│")
    
    print(bottom)
    print(f"\nRating± shows standard error (σ ≈ 400/√n where n is matches played)")
    print(f"W-L-D shows win-loss-draw record")
    print(f"Raw/Cost Avg show mean performance scores across all matches")

def run_tournament(models: List[LLMModel], total_matches: int = None, batch_size: int = 10):
    """Run tournament matches in batches."""
    print("\nSTARTING TOURNAMENT")
    print("==================")
    
    # Initialize prior matches set
    prior_matches = set()
    
    # Track start time
    start_time = time.time()
    
    # Run matches in batches
    batch_num = 1
    total_matches_run = 0
    
    while True:
        print(f"\nBatch {batch_num}")
        print("=" * 20)
        
        # Run a batch of matches
        matches = run_tournament_matches(models, batch_size, prior_matches)
        
        if not matches:
            print("No more valid matches possible.")
            break
            
        # Update prior matches
        for match in matches:
            # Get model names from participants dictionary
            model_names = list(match.participants.values())
            pair = tuple(sorted(model_names))
            prior_matches.add(pair)
            
        # Update counts
        total_matches_run += len(matches)
        batch_num += 1
        
        # Print current standings
        print_model_stats(models)
        
        # Check if we've hit total_matches (if specified)
        if total_matches and total_matches_run >= total_matches:
            print(f"\nReached target of {total_matches} matches.")
            break
            
        # Check if all models have played enough matches
        if all(m.performance["total_matches_played"] >= 50 for m in models):
            print("\nAll models have completed their matches.")
            break
    
    # Calculate and display runtime
    runtime = time.time() - start_time
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60
    
    print(f"\nTournament complete!")
    print(f"Total matches: {total_matches_run}")
    print(f"Runtime: {hours}h {minutes}m {seconds:.1f}s")
    
    # Print final standings
    print("\nFINAL STANDINGS")
    print("===============")
    print_detailed_leaderboard(models)

def main():
    """Main program entry point."""
    print_header()
    
    # Check database connection
    if not check_connection_status():
        print("ERROR: Could not connect to database")
        return
        
    # Initialize models
    print("\nInitializing models...")
    models = initialize_models(MODELS)
    print(f"Initialized {len(models)} models")
    
    # Print initial stats
    print("\nINITIAL RANKINGS")
    print("================")
    print_model_stats(models)
    
    # Run tournament
    run_tournament(models)

if __name__ == "__main__":
    main()