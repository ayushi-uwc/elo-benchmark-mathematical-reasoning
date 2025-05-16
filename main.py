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
from typing import List, Dict, Any

# Import logger configuration first
from logger_config import get_logger, setup_logger

# Import our modules
from database import db, check_connection_status
from models import LLMModel, initialize_models
from model_definitions import MODELS
from tournament import run_tournament_matches, get_top_performers

# Get logger for this module
logger = get_logger(__name__)

# Add near the top of main.py, after logger is set up
logger.info("========== STARTING NEW TOURNAMENT RUN ==========")
# Force flush the log to ensure it's written
try:
    logger.force_flush()
except:
    pass

logger.info("*** IMPORTANT: USING STRICT VERDICT FORMATTING ***")
logger.info("Judges must provide verdicts in the exact format: 'VERDICT: Model {A|B} is superior' or 'VERDICT: This is a tie'")
logger.info("Invalid verdicts will result in judge disqualification and a 10-point ELO penalty")

def print_header():
    """Print application header."""
    header = "\n" + "="*70 + "\n" + " "*25 + "LLM TOURNAMENT SYSTEM" + "\n" + "="*70 + "\n"
    print(header)
    logger.info("Starting LLM Tournament System")
    return header

def print_model_stats(models: List[LLMModel], detailed: bool = False):
    """Print statistics about all models."""
    logger.info("Printing model statistics")
    
    # Sort by raw ELO score
    sorted_models = sorted(models, key=lambda m: -m.elo["raw"]["current"])
    
    stats_header = "\nMODEL RANKINGS:"
    
    # Create expanded table with box-drawing characters to include all requested fields
    header_line = "┌─────┬────────────────────┬──────────┬───────────────┬──────────┬──────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐"
    column_headers = f"│ {'Rank':<3} │ {'Model':<18} │ {'Raw ELO':<8} │ {'Cost-Adj ELO':<13} │ {'W-L':<8} │ {'Matches':<8} │ {'In Cost $':<10} │ {'Out Cost $':<10} │ {'Total Cost':<10} │ {'In Tokens':<10} │ {'Out Tokens':<10} │ {'Total Tokens':<10} │"
    separator = "├─────┼────────────────────┼──────────┼───────────────┼──────────┼──────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤"
    bottom_line = "└─────┴────────────────────┴──────────┴───────────────┴──────────┴──────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘"
    
    # Print to console
    print(stats_header)
    print(header_line)
    print(column_headers)
    print(separator)
    
    # Generate all the stat lines first
    model_stats_lines = []
    model_rank_info = []
    
    # For each model, calculate input/output costs and build table rows
    for i, model in enumerate(sorted_models, 1):
        # Calculate input and output costs for the model
        in_tokens = model.performance['total_input_tokens']
        out_tokens = model.performance['total_output_tokens']
        total_tokens = in_tokens + out_tokens
        
        in_cost = (in_tokens / 1000000) * model.input_cost_per_million
        out_cost = (out_tokens / 1000000) * model.output_cost_per_million
        total_cost = model.performance['total_cost_usd']
        
        # Create stats line with all requested fields
        stats_line = (f"│ {i:<3} │ {model.name:<18} │ {model.elo['raw']['current']:<8.1f} │ "
                      f"{model.elo['cost_adjusted']['current']:<13.1f} │ "
                      f"{model.performance['wins_raw']}-{model.performance['losses_raw']:<6} │ "
                      f"{model.performance['total_matches_played']:<8} │ "
                      f"${in_cost:<9.4f} │ ${out_cost:<9.4f} │ ${total_cost:<9.4f} │ "
                      f"{in_tokens:<10} │ {out_tokens:<10} │ {total_tokens:<10} │")
        
        # Print to console
        print(stats_line)
        
        # Add to our lists for later logging
        model_stats_lines.append(stats_line)
        model_rank_info.append(f"Rank {i}: {model.name} - Raw: {model.elo['raw']['current']:.1f}, "
                              f"Cost-Adj: {model.elo['cost_adjusted']['current']:.1f}, "
                              f"Record: {model.performance['wins_raw']}-{model.performance['losses_raw']}")
    
    # Complete the table in console
    print(bottom_line)
    
    # First output the heading using normal logging
    logger.info("Model Rankings (sorted by raw ELO)")
    
    # Then output the complete table without interruptions using table logger
    logger.table("")  # Empty line for spacing
    logger.table(stats_header)
    logger.table(header_line)
    logger.table(column_headers)
    logger.table(separator)
    
    # Output all table rows at once
    for line in model_stats_lines:
        logger.table(line)
        
    logger.table(bottom_line)
    logger.table("")  # Empty line for spacing
    
    # Then output the individual model info using normal logging
    for info in model_rank_info:
        logger.info(info)
    
    if detailed:
        logger.info("Printing detailed model information")
        print("\nDETAILED MODEL INFORMATION:")
        for model in sorted_models:
            print(f"\n{model.name}:")
            print(f"  Provider: {model.provider}")
            print(f"  Model ID: {model.model_id}")
            print(f"  Input cost: ${model.input_cost_per_million}/million tokens")
            print(f"  Output cost: ${model.output_cost_per_million}/million tokens")
            if hasattr(model, 'pricing_source') and model.pricing_source:
                print(f"  Pricing source: {model.pricing_source}")
            print(f"  ELO rating: {model.elo['raw']['current']:.1f} (raw), {model.elo['cost_adjusted']['current']:.1f} (cost-adjusted)")
            print(f"  Matches played: {model.performance['total_matches_played']}")
            print(f"  Raw ELO: {model.elo['raw']['current']:.1f} (started at {model.elo['raw']['initial']})")
            print(f"  Cost-Adjusted ELO: {model.elo['cost_adjusted']['current']:.1f}")
            print(f"  Record: {model.performance['wins_raw']}-{model.performance['losses_raw']}")
            print(f"  Total input tokens: {model.performance['total_input_tokens']}")
            print(f"  Total output tokens: {model.performance['total_output_tokens']}")
            print(f"  Avg. input tokens per response: {model.performance['total_input_tokens'] / model.performance['total_matches_played'] if model.performance['total_matches_played'] > 0 else 0:.1f}")
            print(f"  Avg. output tokens per response: {model.performance['total_output_tokens'] / model.performance['total_matches_played'] if model.performance['total_matches_played'] > 0 else 0:.1f}")
            print(f"  Avg. tokens per response: {model.avg_tokens_per_response:.1f}")
            print(f"  Avg. cost per response: ${model.avg_cost_per_response_usd:.4f}")
            print(f"  Last updated: {model.metadata['last_updated']}")
            
            # Log match history
            logger.info(f"Model {model.name} detailed stats:")
            logger.info(f"  Match history: {model.performance['wins_raw']}-{model.performance['losses_raw']}")
            logger.info(f"  Matches played: {len(model.match_ids['played'])}")
            logger.info(f"  Judgements made: {len(model.match_ids['judged'])}")
            logger.info(f"  Cases generated: {len(model.match_ids['cases_generated'])}")
            logger.info(f"  Questions generated: {len(model.match_ids['questions_generated'])}")

def run_tournament(models: List[LLMModel], total_matches: int = None, batch_size: int = 10):
    """
    Run tournament to ensure each model plays exactly 20 matches.
    
    Args:
        models: List of all models
        total_matches: Optional safety limit for total matches (automatically calculated if None)
        batch_size: Number of matches per batch for status reporting
    
    Returns:
        List of all matches executed
    """
    # Calculate required matches to get 20 per model
    target_matches_per_model = 20
    
    # Calculate a reasonable max_matches to use as safety limit if not provided
    # This would be at most (n*20)/2 matches where n is the number of models
    if total_matches is None:
        theoretical_max = (len(models) * target_matches_per_model) // 2
        # Add a small buffer for safety
        total_matches = theoretical_max + len(models)
    
    logger.info(f"Starting tournament with target of {target_matches_per_model} matches per model")
    logger.info(f"Maximum total matches limit: {total_matches}")
    logger.info(f"Status updates will be provided every {batch_size} matches")
    
    all_matches = []
    prior_matches = set()
    completed_matches = 0
    batch_num = 1
    
    # Get current matches played per model
    model_matches = {model.name: model.performance["total_matches_played"] for model in models}
    logger.info("Current match counts per model:")
    for name, count in sorted(model_matches.items()):
        logger.info(f"  {name}: {count}/{target_matches_per_model}")
    
    # Calculate remaining matches needed
    remaining_matches = sum(max(0, target_matches_per_model - count) for count in model_matches.values())
    logger.info(f"Total remaining model-matches needed: {remaining_matches}")
    
    # Continue until all models have played enough matches
    current_batch_size = min(batch_size, total_matches)
    while (any(model_matches[m.name] < target_matches_per_model for m in models) and 
           completed_matches < total_matches):
        
        logger.info(f"Beginning batch {batch_num} with max {current_batch_size} matches")
        print(f"\nProcessing batch {batch_num} (max {current_batch_size} matches)...")
        
        # Run a batch of matches with a safety limit
        matches = run_tournament_matches(
            models, 
            max_matches=current_batch_size,
            prior_matches=prior_matches
        )
        
        # If no matches were created, break out of the loop
        if not matches:
            logger.warning(f"No valid matches could be created in batch {batch_num}. Ending tournament.")
            break
            
        all_matches.extend(matches)
        completed_matches += len(matches)
        
        # Update prior matches for next batch
        for match in matches:
            model_names = []
            for model_id in match.participants.values():
                if model_id:
                    for model in models:
                        if model.model_id == model_id:
                            model_names.append(model.name)
                            break
            if len(model_names) == 2:
                prior_matches.add(tuple(sorted(model_names)))
        
        # Update match counts per model
        model_matches = {model.name: model.performance["total_matches_played"] for model in models}
        
        # Status reporting
        logger.info(f"Completed batch {batch_num} with {len(matches)} matches")
        logger.info(f"Total matches so far: {completed_matches}/{total_matches} (safety limit)")
        
        # Match count table
        logger.info("Current match counts per model:")
        models_at_target = 0
        for name, count in sorted(model_matches.items()):
            status = "✓" if count >= target_matches_per_model else f"{target_matches_per_model-count} more needed"
            logger.info(f"  {name}: {count}/{target_matches_per_model} ({status})")
            if count >= target_matches_per_model:
                models_at_target += 1
        
        logger.info(f"Progress: {models_at_target}/{len(models)} models have reached target")
        logger.info(f"Total unique matchups so far: {len(prior_matches)}")
        
        # Show updated rankings after each batch
        print_model_stats(models)
        
        # Update counters
        batch_num += 1
        
        # Pause between batches if there are more to go
        if any(model_matches[m.name] < target_matches_per_model for m in models):
            print(f"\nPausing between batches (3 seconds)...")
            time.sleep(3)
    
    # Final report
    all_at_target = all(model_matches[m.name] >= target_matches_per_model for m in models)
    if all_at_target:
        logger.info(f"SUCCESS: All models have played at least {target_matches_per_model} matches")
    else:
        models_below_target = [name for name, count in model_matches.items() if count < target_matches_per_model]
        logger.info(f"INCOMPLETE: {len(models_below_target)} models did not reach {target_matches_per_model} matches")
        logger.info(f"Models below target: {', '.join(models_below_target)}")
    
    # Log tournament completion stats
    possible_pairs = len(models) * (len(models) - 1) // 2
    coverage_pct = (len(prior_matches) / possible_pairs * 100) if possible_pairs > 0 else 0
    logger.info(f"Tournament complete: {len(all_matches)} total matches executed")
    logger.info(f"Matchup coverage: {len(prior_matches)}/{possible_pairs} pairs ({coverage_pct:.1f}%)")
    
    return all_matches

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM Tournament System")
    parser.add_argument("--max-matches", type=int, default=None, 
                      help="Maximum total number of matches to run (safety limit)")
    parser.add_argument("--batch-size", type=int, default=5, 
                      help="Number of matches per batch for status updates")
    parser.add_argument("--stats", action="store_true", 
                      help="Show detailed model statistics")
    parser.add_argument("--stats-only", action="store_true", 
                      help="Only show statistics, don't run tournament")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      default="INFO", help="Set logging level")
    parser.add_argument("--install-deps", action="store_true",
                      help="Check and install dependencies")
    
    args = parser.parse_args()
    
    # Set logging level based on command line arg
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)
    logger.info(f"Log level set to {args.log_level}")
    
    print_header()
    
    # Check for dependencies
    if args.install_deps:
        logger.info("Checking for dependencies...")
        try:
            import pkg_resources

            # Define required packages
            required_packages = [
                "sentence-transformers>=2.2.0",
                "torch>=1.11.0"
            ]
            
            # Check and install missing packages
            import subprocess
            import sys
            
            for package in required_packages:
                try:
                    pkg_resources.require(package)
                    logger.info(f"Package {package} is already installed.")
                except pkg_resources.DistributionNotFound:
                    logger.info(f"Installing missing package: {package}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"Successfully installed {package}")
            
            logger.info("All dependencies are installed.")
        except Exception as e:
            logger.error(f"Error checking/installing dependencies: {str(e)}")
    
    # Check database connection
    logger.info("Checking database connection")
    if not check_connection_status():
        logger.critical("Database connection failed. Exiting...")
        print("Database connection failed. Exiting...")
        sys.exit(1)
    
    # Initialize models
    logger.info("Initializing models from model definitions")
    print("\nInitializing models...")
    all_models = initialize_models(MODELS)
    logger.info(f"Initialized {len(all_models)} models")
    
    # Show initial statistics
    print_model_stats(all_models, detailed=args.stats)
    
    # Exit if stats-only mode
    if args.stats_only:
        logger.info("Stats-only mode, exiting without running tournament")
        print("\nStatistics display complete. Exiting...")
        return
    
    # Run tournament to ensure each model plays exactly 20 matches
    logger.info("Running tournament with target of 20 matches per model")
    print("\nRunning tournament to ensure each model plays exactly 20 matches...")
    
    matches = run_tournament(all_models, args.max_matches, args.batch_size)
    
    # Show final results
    total_matches = len(matches)
    logger.info(f"Tournament complete: {total_matches} matches executed")
    print("\nTournament complete!")
    print(f"Executed {total_matches} matches")
    
    # Final statistics
    print("\nFinal Rankings:")
    print_model_stats(all_models, detailed=args.stats)
    
    logger.info("Tournament system execution complete.")
    print("\nTournament system execution complete.")
    
    # Make sure logs are flushed before exit
    try:
        logger.force_flush()
        logging.shutdown()
    except:
        pass

if __name__ == "__main__":
    main()