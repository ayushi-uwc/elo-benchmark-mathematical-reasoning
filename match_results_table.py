#!/usr/bin/env python3
"""
Match Results Table Generator

This script extracts match data from both the MongoDB database and log files,
to generate a clean table showing which models played against each other and 
their win-loss-draw records (W-L-D).

The script creates:
1. A detailed table of all matchups
2. A model-vs-model table for each model
3. A summary table in the format: LLM 1, Played with (LLM2), W-L-D, Played with (LLM3), W-L-D, etc.
4. CSV exports of the results
"""

import os
import re
import glob
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

# Import needed modules from the codebase
from database import db, check_connection_status
from models import LLMModel, initialize_models
from model_definitions import MODELS
from matches import Match

def get_model_name_from_id(model_id: str, model_map: Dict[str, str]) -> str:
    """
    Get model name from model_id using the model map.
    
    Args:
        model_id: The model ID to look up
        model_map: Dictionary mapping model IDs to model names
        
    Returns:
        The model name if found, otherwise returns the model_id
    """
    return model_map.get(model_id, model_id)

def parse_log_files(log_dir: str = "logs", model_map: Dict[str, str] = None) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Parse log files to extract match results.
    
    Args:
        log_dir: Directory containing log files
        model_map: Dictionary mapping model IDs to model names
        
    Returns:
        Dictionary of match results, keyed by model pairs
    """
    if model_map is None:
        model_map = {}
        
    print(f"\nParsing log files from {log_dir}...")
    
    # Dictionary to store match results: {(model_a, model_b): {"wins_a": 0, "wins_b": 0, "draws": 0}}
    log_match_results = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})
    
    # Track processed match IDs to avoid duplicates
    processed_match_ids = set()
    
    # Get list of log files sorted by modification time (newest first)
    log_files = sorted(
        glob.glob(os.path.join(log_dir, "*.log")), 
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )
    
    if not log_files:
        print("No log files found.")
        return log_match_results
        
    print(f"Found {len(log_files)} log files.")
    
    # Regex patterns to extract match information
    match_id_pattern = r"Match ID: (match_[a-zA-Z0-9]+)"
    participants_pattern = r"Participants: \{'model_a': '([^']+)', 'model_b': '([^']+)'\}"
    results_pattern = r"Raw scores: ([A-Za-z0-9\s\-\.]+) ([0-9\.]+) - ([0-9\.]+) ([A-Za-z0-9\s\-\.]+)"
    
    for log_file in log_files:
        print(f"Processing {os.path.basename(log_file)}...")
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                
                # Find all matches in this log file
                match_id_matches = re.finditer(match_id_pattern, log_content)
                
                for match_id_match in match_id_matches:
                    match_id = match_id_match.group(1)
                    
                    # Skip if already processed
                    if match_id in processed_match_ids:
                        continue
                        
                    # Get match start position
                    match_start_pos = match_id_match.start()
                    
                    # Look for participants in the next 1000 characters
                    participants_search_text = log_content[match_start_pos:match_start_pos + 1000]
                    participants_match = re.search(participants_pattern, participants_search_text)
                    
                    if not participants_match:
                        continue
                        
                    model_a_id = participants_match.group(1)
                    model_b_id = participants_match.group(2)
                    
                    # Get model names
                    model_a_name = get_model_name_from_id(model_a_id, model_map)
                    model_b_name = get_model_name_from_id(model_b_id, model_map)
                    
                    # Create a consistent key for this matchup (alphabetically sorted)
                    model_pair = tuple(sorted([model_a_name, model_b_name]))
                    
                    # Look for results in the next 20000 characters after participants
                    result_search_text = log_content[match_start_pos:match_start_pos + 20000]
                    result_match = re.search(results_pattern, result_search_text)
                    
                    if result_match:
                        # Extract names and scores
                        model_name_a = result_match.group(1).strip()
                        score_a = float(result_match.group(2))
                        score_b = float(result_match.group(3))
                        model_name_b = result_match.group(4).strip()
                        
                        # Verify names match expected models
                        name_a_matches = model_a_name in model_name_a or model_a_name in model_a_name
                        name_b_matches = model_b_name in model_name_b or model_b_name in model_b_name
                        
                        if not (name_a_matches and name_b_matches):
                            # Try swapping the names
                            if model_a_name in model_name_b and model_b_name in model_name_a:
                                # Names are swapped, swap scores as well
                                score_a, score_b = score_b, score_a
                            else:
                                # Names don't match, skip this match
                                continue
                        
                        # Determine result (using threshold of 0.51 to account for floating point)
                        if score_a > 0.51:
                            # Model A won
                            if model_pair[0] == model_a_name:
                                log_match_results[model_pair]["wins_a"] += 1
                            else:
                                log_match_results[model_pair]["wins_b"] += 1
                        elif score_b > 0.51:
                            # Model B won
                            if model_pair[0] == model_b_name:
                                log_match_results[model_pair]["wins_a"] += 1
                            else:
                                log_match_results[model_pair]["wins_b"] += 1
                        else:
                            # Draw
                            log_match_results[model_pair]["draws"] += 1
                            
                        # Mark as processed
                        processed_match_ids.add(match_id)
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
            continue
    
    print(f"Extracted {len(log_match_results)} matchups from log files.")
    print(f"Processed {len(processed_match_ids)} unique matches.")
    
    return log_match_results

def merge_results(db_results: Dict, log_results: Dict) -> Dict:
    """
    Merge results from database and log files, prioritizing database results.
    
    Args:
        db_results: Match results from the database
        log_results: Match results from log files
        
    Returns:
        Merged results dictionary
    """
    merged_results = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})
    
    # Add all database results
    for model_pair, results in db_results.items():
        merged_results[model_pair] = results.copy()
    
    # Add log results only for model pairs not in database
    for model_pair, results in log_results.items():
        if model_pair not in merged_results:
            merged_results[model_pair] = results.copy()
    
    return merged_results

def format_wld_for_csv(wins: int, losses: int, draws: int) -> str:
    """
    Format the W-L-D string with an apostrophe prefix to prevent interpretation as a date in spreadsheets.
    
    Args:
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        
    Returns:
        Formatted W-L-D string with apostrophe prefix
    """
    return f"'{wins}-{losses}-{draws}"

def create_matchup_tables(matchup_results: Dict, models: List[LLMModel]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create DataFrame tables from matchup results.
    
    Args:
        matchup_results: Dictionary of match results
        models: List of LLMModel objects
        
    Returns:
        Tuple of (detailed_df, pivot_df)
    """
    # Create a DataFrame for the detailed results
    results_data = []
    
    for model_pair, results in matchup_results.items():
        model_a, model_b = model_pair
        wins_a = results["wins_a"]
        wins_b = results["wins_b"]
        draws = results["draws"]
        total = wins_a + wins_b + draws
        
        # Format W-L-D with apostrophe prefix for CSV export
        wld_formatted = format_wld_for_csv(wins_a, wins_b, draws)
        
        results_data.append({
            "Model A": model_a,
            "Model B": model_b,
            "Wins A": wins_a,
            "Wins B": wins_b,
            "Draws": draws,
            "W-L-D": wld_formatted,
            "Total Matches": total
        })
    
    # Convert to DataFrame and sort
    detailed_df = pd.DataFrame(results_data)
    detailed_df.sort_values(by=["Model A", "Model B"], inplace=True)
    
    # Create a pivoted table for each model showing all its matchups
    model_matchups = {}
    
    for model in models:
        model_name = model.name
        model_matchups[model_name] = {}
    
    for model_pair, results in matchup_results.items():
        model_a, model_b = model_pair
        wins_a = results["wins_a"]
        wins_b = results["wins_b"]
        draws = results["draws"]
        
        # Add results from model_a's perspective - with apostrophe prefix for CSV
        if model_a == model_pair[0]:
            wld_formatted = format_wld_for_csv(wins_a, wins_b, draws)
        else:
            wld_formatted = format_wld_for_csv(wins_b, wins_a, draws)
        model_matchups[model_a][model_b] = wld_formatted
        
        # Add results from model_b's perspective - with apostrophe prefix for CSV
        if model_b == model_pair[0]:
            wld_formatted = format_wld_for_csv(wins_a, wins_b, draws)
        else:
            wld_formatted = format_wld_for_csv(wins_b, wins_a, draws)
        model_matchups[model_b][model_a] = wld_formatted
    
    # Create pivot DataFrame
    pivot_data = []
    for model_name, matchups in model_matchups.items():
        row = {"Model": model_name}
        for opponent, record in matchups.items():
            row[f"vs {opponent}"] = record
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    return detailed_df, pivot_df, model_matchups

def print_results(detailed_df: pd.DataFrame, pivot_df: pd.DataFrame, model_matchups: Dict):
    """
    Print formatted results to the console.
    
    Args:
        detailed_df: DataFrame with detailed results
        pivot_df: DataFrame with pivot table
        model_matchups: Dictionary of model matchups
    """
    # For console display, we strip the apostrophe prefix
    detailed_df_display = detailed_df.copy()
    detailed_df_display["W-L-D"] = detailed_df_display["W-L-D"].str.replace("'", "")
    
    # Print tabular results
    print("\nMatchup Results:")
    print(detailed_df_display.to_string(index=False))
    
    # Print detailed model-vs-model table
    print("\nDetailed Model-vs-Model Results:")
    
    for model_name, matchups in sorted(model_matchups.items()):
        print(f"\n{model_name}:")
        for opponent, record in sorted(matchups.items()):
            # Strip apostrophe prefix for console display
            clean_record = record.replace("'", "")
            print(f"  vs {opponent}: {clean_record}")
    
    # Create a summary table: LLM 1, Played with (LLM2), W-L-D, Played with (LLM3), W-L-D, etc.
    print("\nSummary Table:")
    print(f"{'Model':<30} {'Matchups'}")
    print("-" * 100)
    
    for model_name, matchups in sorted(model_matchups.items()):
        summary = []
        for opponent, record in sorted(matchups.items()):
            # Strip apostrophe prefix for console display
            clean_record = record.replace("'", "")
            summary.append(f"Played with ({opponent}), {clean_record}")
        
        matchup_str = ", ".join(summary)
        print(f"{model_name:<30} {matchup_str}")

def save_results(detailed_df: pd.DataFrame, pivot_df: pd.DataFrame):
    """
    Save results to CSV files.
    
    Args:
        detailed_df: DataFrame with detailed results
        pivot_df: DataFrame with pivot table
    """
    detailed_df.to_csv("matchup_results.csv", index=False)
    print("\nSaved results to matchup_results.csv")
    
    pivot_df.to_csv("model_matchups_pivot.csv", index=False)
    print("Saved pivot table to model_matchups_pivot.csv")

def generate_matchup_table():
    """
    Generate a table showing all model matchups and their results.
    Combines data from both the database and log files.
    """
    print("Connecting to database...")
    if not check_connection_status():
        print("Failed to connect to database. Exiting.")
        return
    
    print("Initializing models...")
    models = initialize_models(MODELS)
    
    # Create model_id to model_name mapping
    model_map = {model.model_id: model.name for model in models}
    
    print(f"Found {len(models)} models")
    
    # Get results from database
    print("Loading matches from database...")
    db_matches = Match.get_all_matches()
    print(f"Found {len(db_matches)} matches in database")
    
    # Create a dictionary to store match results from database
    db_matchup_results = defaultdict(lambda: {"wins_a": 0, "wins_b": 0, "draws": 0})
    
    # Process database matches
    for match in db_matches:
        # Extract model IDs from participants
        model_a_id = match.participants.get("model_a")
        model_b_id = match.participants.get("model_b")
        
        if not model_a_id or not model_b_id:
            continue
        
        # Get model names
        model_a_name = get_model_name_from_id(model_a_id, model_map)
        model_b_name = get_model_name_from_id(model_b_id, model_map)
        
        # Create a consistent key for this matchup (alphabetically sorted)
        model_pair = tuple(sorted([model_a_name, model_b_name]))
        
        # Check if this match has judge votes in its judgment
        if "judges" in match.judgment and match.judgment["judges"]:
            # Count individual judge votes instead of using the final score
            for judge_entry in match.judgment["judges"]:
                vote = judge_entry.get("vote")
                vote_type = judge_entry.get("vote_type", "win")  # Default to "win" for backward compatibility
                
                if vote_type == "win":
                    if vote == model_a_id:
                        # Judge voted for model A
                        if model_pair[0] == model_a_name:
                            db_matchup_results[model_pair]["wins_a"] += 1
                        else:
                            db_matchup_results[model_pair]["wins_b"] += 1
                    elif vote == model_b_id:
                        # Judge voted for model B
                        if model_pair[0] == model_b_name:
                            db_matchup_results[model_pair]["wins_a"] += 1
                        else:
                            db_matchup_results[model_pair]["wins_b"] += 1
                elif vote_type == "draw":
                    # Judge declared a draw
                    db_matchup_results[model_pair]["draws"] += 1
        else:
            # Fallback to using raw scores if judge votes not available
            raw_scores = match.judgment.get("raw_score", {})
            
            if model_a_id in raw_scores and model_b_id in raw_scores:
                score_a = raw_scores[model_a_id]
                score_b = raw_scores[model_b_id]
                
                # Determine result (using threshold of 0.51 to account for floating point)
                if score_a > 0.51:
                    # Model A won
                    if model_pair[0] == model_a_name:
                        db_matchup_results[model_pair]["wins_a"] += 1
                    else:
                        db_matchup_results[model_pair]["wins_b"] += 1
                elif score_b > 0.51:
                    # Model B won
                    if model_pair[0] == model_b_name:
                        db_matchup_results[model_pair]["wins_a"] += 1
                    else:
                        db_matchup_results[model_pair]["wins_b"] += 1
                else:
                    # Draw
                    db_matchup_results[model_pair]["draws"] += 1
    
    # Get results from log files
    log_matchup_results = parse_log_files(log_dir="logs", model_map=model_map)
    
    # Merge results from both sources
    print("\nMerging results from database and logs...")
    merged_results = merge_results(db_matchup_results, log_matchup_results)
    print(f"Final merged dataset contains {len(merged_results)} unique matchups")
    
    # Create tables
    detailed_df, pivot_df, model_matchups = create_matchup_tables(merged_results, models)
    
    # Print results
    print_results(detailed_df, pivot_df, model_matchups)
    
    # Save results
    save_results(detailed_df, pivot_df)

if __name__ == "__main__":
    generate_matchup_table() 