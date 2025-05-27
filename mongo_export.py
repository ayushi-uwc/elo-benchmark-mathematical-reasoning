#!/usr/bin/env python3
"""
MongoDB Export Script

This script exports data from MongoDB collections (models and matches) to CSV files.
"""

import csv
import pandas as pd
from typing import Dict, List, Any

# Import needed modules from the codebase
from database import db, check_connection_status
from config import MODELS_COLLECTION, MATCHES_COLLECTION

def export_models_collection():
    """
    Export models collection to CSV file.
    """
    print(f"Exporting {MODELS_COLLECTION} collection to CSV...")
    
    # Check database connection
    if not check_connection_status():
        print("Failed to connect to database. Exiting.")
        return
    
    # Get all models
    models_data = db.get_all_models()
    print(f"Found {len(models_data)} models in database")
    
    if not models_data:
        print("No models found in database.")
        return
    
    # Flatten nested fields for CSV export
    flattened_models = []
    for model in models_data:
        flat_model = {}
        
        # Basic fields
        flat_model['name'] = model.get('name', '')
        flat_model['model_id'] = model.get('model_id', '')
        flat_model['provider'] = model.get('provider', '')
        flat_model['cost_per_million'] = model.get('cost_per_million', 0)
        flat_model['input_cost_per_million'] = model.get('input_cost_per_million', 0)
        
        # ELO ratings
        if 'elo' in model:
            if 'raw' in model['elo']:
                flat_model['raw_elo_initial'] = model['elo']['raw'].get('initial', 1500)
                flat_model['raw_elo_current'] = model['elo']['raw'].get('current', 1500)
            
            if 'cost_adjusted' in model['elo']:
                flat_model['cost_elo_initial'] = model['elo']['cost_adjusted'].get('initial', 1500)
                flat_model['cost_elo_current'] = model['elo']['cost_adjusted'].get('current', 1500)
        
        # Performance metrics
        if 'performance' in model:
            perf = model['performance']
            flat_model['total_matches_played'] = perf.get('total_matches_played', 0)
            flat_model['wins_raw'] = perf.get('wins_raw', 0)
            flat_model['losses_raw'] = perf.get('losses_raw', 0)
            flat_model['draws_raw'] = perf.get('draws_raw', 0)
            flat_model['total_tokens_used'] = perf.get('total_tokens_used', 0)
            flat_model['total_cost_usd'] = perf.get('total_cost_usd', 0)
            
            # Score history
            if 'score_history' in perf:
                flat_model['avg_raw_score'] = perf['score_history'].get('avg_raw_score', 0)
                flat_model['avg_adjusted_score'] = perf['score_history'].get('avg_adjusted_score', 0)
        
        # Match IDs (convert lists to comma-separated strings)
        if 'match_ids' in model:
            match_ids = model['match_ids']
            flat_model['played_matches'] = ','.join(match_ids.get('played', []))
            flat_model['judged_matches'] = ','.join(match_ids.get('judged', []))
            flat_model['cases_generated'] = ','.join(match_ids.get('cases_generated', []))
            flat_model['questions_generated'] = ','.join(match_ids.get('questions_generated', []))
        
        flattened_models.append(flat_model)
    
    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(flattened_models)
    csv_filename = f"{MODELS_COLLECTION}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Exported {len(flattened_models)} models to {csv_filename}")
    
    # Print column names
    print(f"Columns in {csv_filename}:")
    for col in df.columns:
        print(f"  - {col}")

def export_matches_collection():
    """
    Export matches collection to CSV file.
    """
    print(f"\nExporting {MATCHES_COLLECTION} collection to CSV...")
    
    # Check database connection
    if not check_connection_status():
        print("Failed to connect to database. Exiting.")
        return
    
    # Get all matches
    matches_data = db.get_matches({})
    print(f"Found {len(matches_data)} matches in database")
    
    if not matches_data:
        print("No matches found in database.")
        return
    
    # Flatten nested fields for CSV export
    flattened_matches = []
    for match in matches_data:
        flat_match = {}
        
        # Basic fields
        flat_match['_id'] = match.get('_id', '')
        flat_match['timestamp'] = match.get('timestamp', '')
        
        # Participants
        if 'participants' in match:
            flat_match['model_a'] = match['participants'].get('model_a', '')
            flat_match['model_b'] = match['participants'].get('model_b', '')
        
        # Prompt info
        if 'prompt' in match:
            prompt = match['prompt']
            flat_match['case_generator_id'] = prompt.get('case_generator_id', '')
            flat_match['question_generator_id'] = prompt.get('question_generator_id', '')
            
            # Store case text and question text in separate files if needed
            # They can be very long, so we'll exclude them from the main CSV
            flat_match['has_case_text'] = bool(prompt.get('case_text', ''))
            flat_match['has_question_text'] = bool(prompt.get('question_text', ''))
        
        # Responses
        if 'responses' in match:
            for model_id, response in match['responses'].items():
                prefix = 'model_a' if model_id == flat_match.get('model_a', '') else 'model_b'
                
                # Token usage and costs
                flat_match[f'{prefix}_tokens'] = response.get('tokens', 0)
                flat_match[f'{prefix}_input_tokens'] = response.get('input_tokens', 0)
                flat_match[f'{prefix}_output_tokens'] = response.get('output_tokens', 0)
                flat_match[f'{prefix}_cost_usd'] = response.get('cost_usd', 0)
                flat_match[f'{prefix}_input_cost_usd'] = response.get('input_cost_usd', 0)
                flat_match[f'{prefix}_output_cost_usd'] = response.get('output_cost_usd', 0)
                
                # ELO before and after
                flat_match[f'{prefix}_elo_raw_before'] = response.get('elo_raw_before', 0)
                flat_match[f'{prefix}_elo_raw_after'] = response.get('elo_raw_after', 0)
                flat_match[f'{prefix}_elo_cost_before'] = response.get('elo_cost_before', 0)
                flat_match[f'{prefix}_elo_cost_after'] = response.get('elo_cost_after', 0)
                
                # Store response text in separate files if needed
                flat_match[f'{prefix}_has_response'] = bool(response.get('text', ''))
        
        # Judgment
        if 'judgment' in match:
            judgment = match['judgment']
            
            # Raw scores
            if 'raw_score' in judgment:
                for model_id, score in judgment['raw_score'].items():
                    prefix = 'model_a' if model_id == flat_match.get('model_a', '') else 'model_b'
                    flat_match[f'{prefix}_raw_score'] = score
            
            # Cost-adjusted scores
            if 'cost_adjusted_score' in judgment:
                for model_id, score in judgment['cost_adjusted_score'].items():
                    prefix = 'model_a' if model_id == flat_match.get('model_a', '') else 'model_b'
                    flat_match[f'{prefix}_cost_adjusted_score'] = score
            
            # Judge votes
            if 'judges' in judgment:
                # Count votes for each model and draws
                votes_a = 0
                votes_b = 0
                draws = 0
                
                for judge_entry in judgment['judges']:
                    vote = judge_entry.get('vote', '')
                    vote_type = judge_entry.get('vote_type', 'win')
                    
                    if vote_type == 'draw':
                        draws += 1
                    elif vote == flat_match.get('model_a', ''):
                        votes_a += 1
                    elif vote == flat_match.get('model_b', ''):
                        votes_b += 1
                
                flat_match['votes_a'] = votes_a
                flat_match['votes_b'] = votes_b
                flat_match['votes_draw'] = draws
                flat_match['total_votes'] = votes_a + votes_b + draws
                flat_match['wld_a'] = f"'{votes_a}-{votes_b}-{draws}"  # Apostrophe prefix to prevent date interpretation
                
                # Store number of judges
                flat_match['judge_count'] = len(judgment['judges'])
                
                # Store judge IDs as comma-separated string
                judge_ids = [judge.get('judge_id', '') for judge in judgment['judges']]
                flat_match['judge_ids'] = ','.join(judge_ids)
        
        # Meta information
        if 'meta' in match:
            meta = match['meta']
            flat_match['stratum'] = meta.get('stratum', '')
            flat_match['matchup_index'] = meta.get('matchup_index', 0)
            flat_match['judgment_method'] = meta.get('judgment_method', '')
            flat_match['cost_temperature'] = meta.get('cost_temperature', 0)
            flat_match['elo_k_factor'] = meta.get('elo_k_factor', 0)
        
        flattened_matches.append(flat_match)
    
    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(flattened_matches)
    csv_filename = f"{MATCHES_COLLECTION}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Exported {len(flattened_matches)} matches to {csv_filename}")
    
    # Print column names
    print(f"Columns in {csv_filename}:")
    for col in df.columns:
        print(f"  - {col}")

def export_judge_votes():
    """
    Export detailed judge votes to a separate CSV file.
    """
    print("\nExporting judge votes to CSV...")
    
    # Check database connection
    if not check_connection_status():
        print("Failed to connect to database. Exiting.")
        return
    
    # Get all matches
    matches_data = db.get_matches({})
    
    if not matches_data:
        print("No matches found in database.")
        return
    
    # Extract judge votes
    all_votes = []
    for match in matches_data:
        match_id = match.get('_id', '')
        
        # Get participant models
        model_a = match.get('participants', {}).get('model_a', '')
        model_b = match.get('participants', {}).get('model_b', '')
        
        # Get judges' votes
        if 'judgment' in match and 'judges' in match['judgment']:
            for judge_entry in match['judgment']['judges']:
                vote_entry = {
                    'match_id': match_id,
                    'model_a': model_a,
                    'model_b': model_b,
                    'judge_id': judge_entry.get('judge_id', ''),
                    'judge_name': judge_entry.get('judge_name', ''),
                    'judge_elo': judge_entry.get('elo_at_judgment', 0),
                    'vote': judge_entry.get('vote', ''),
                    'vote_type': judge_entry.get('vote_type', 'win'),
                    'weight': judge_entry.get('weight', 0),
                }
                
                # Determine vote direction
                if vote_entry['vote_type'] == 'draw':
                    vote_entry['voted_for'] = 'draw'
                elif vote_entry['vote'] == model_a:
                    vote_entry['voted_for'] = 'model_a'
                elif vote_entry['vote'] == model_b:
                    vote_entry['voted_for'] = 'model_b'
                else:
                    vote_entry['voted_for'] = 'unknown'
                
                all_votes.append(vote_entry)
    
    # Convert to DataFrame and export to CSV
    if all_votes:
        df = pd.DataFrame(all_votes)
        csv_filename = "judge_votes.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Exported {len(all_votes)} judge votes to {csv_filename}")
        
        # Print column names
        print(f"Columns in {csv_filename}:")
        for col in df.columns:
            print(f"  - {col}")
    else:
        print("No judge votes found.")

if __name__ == "__main__":
    print("MongoDB Export Script")
    print("=====================")
    
    # Export models collection
    export_models_collection()
    
    # Export matches collection
    export_matches_collection()
    
    # Export judge votes
    export_judge_votes()
    
    print("\nExport complete!") 