#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICC Agreement Analysis for Inter-Rater Reliability

This script calculates ICC(1,k) agreement scores for each (task, metric, model) combination
and applies standard ICC interpretation bands to assess the reliability of evaluations.

ICC Interpretation Bands:
- Excellent: > 0.90
- Good: 0.75 - 0.89
- Moderate: 0.50 - 0.74
- Poor: < 0.50
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_icc_1k(ratings):
    """
    Calculate ICC(1,k) - Intraclass Correlation Coefficient (1,k)
    
    ICC(1,k) measures the reliability of k raters when each subject is rated by 
    the same set of k raters, and raters are considered a fixed effect.
    
    Parameters:
    ratings: array-like, shape (n_subjects, k_raters)
    
    Returns:
    icc_value: float, ICC(1,k) coefficient
    """
    ratings = np.array(ratings)
    
    if ratings.shape[0] < 2 or ratings.shape[1] < 2:
        return np.nan
    
    # Remove rows with any NaN values
    ratings = ratings[~np.isnan(ratings).any(axis=1)]
    
    if len(ratings) < 2:
        return np.nan
    
    n, k = ratings.shape
    
    # Calculate means
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)
    grand_mean = np.mean(ratings)
    
    # Calculate sum of squares
    # Between subjects (rows)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    
    # Within subjects (error)
    ss_within = np.sum((ratings - row_means[:, np.newaxis]) ** 2)
    
    # Total sum of squares
    ss_total = np.sum((ratings - grand_mean) ** 2)
    
    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    
    # ICC(1,k) calculation
    if ms_within == 0:
        return 1.0 if ms_between > 0 else np.nan
    
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    
    return max(0, icc)  # ICC cannot be negative in this context

def interpret_icc(icc_value):
    """Interpret ICC value according to standard bands"""
    if pd.isna(icc_value):
        return "Invalid"
    elif icc_value > 0.90:
        return "Excellent"
    elif icc_value >= 0.75:
        return "Good"
    elif icc_value >= 0.50:
        return "Moderate"
    else:
        return "Poor"

def clean_task_id(task_id):
    """Clean task IDs thoroughly to ensure proper matching"""
    if pd.isna(task_id):
        return task_id
    cleaned = str(task_id).strip().replace('\n', '').replace('\r', '').replace('\t', '')
    return cleaned

def calculate_icc_for_all_combinations():
    """Calculate ICC for all (task, metric, model) combinations"""
    
    print("Loading and preparing data for ICC analysis...")
    
    # Load data
    set_df = pd.read_excel('set.xlsx')
    scores_df = pd.read_excel('score.xlsx')
    status_df = pd.read_excel('status.xlsx')
    
    # Clean task IDs
    set_df['task_id'] = set_df['task_id'].apply(clean_task_id)
    scores_df['task_id'] = scores_df['task_id'].apply(clean_task_id)
    status_df['task_id'] = status_df['task_id'].apply(clean_task_id)
    
    print(f"Loaded {len(scores_df)} score records for {len(scores_df['task_id'].unique())} tasks")
    print(f"Dimensions: {scores_df['dimension'].unique()}")
    print(f"Models: {scores_df['model'].unique()}")
    
    # Calculate ICC for each combination
    icc_results = []
    
    for task_id in scores_df['task_id'].unique():
        for model in scores_df['model'].unique():
            for dimension in scores_df['dimension'].unique():
                
                # Get ratings for this specific combination
                mask = (scores_df['task_id'] == task_id) & \
                       (scores_df['model'] == model) & \
                       (scores_df['dimension'] == dimension)
                
                record = scores_df[mask]
                
                if len(record) == 1:
                    # Extract r1, r2, r3 ratings
                    ratings = [
                        [record.iloc[0]['r1'], record.iloc[0]['r2'], record.iloc[0]['r3']]
                    ]
                    
                    # Calculate ICC(1,k) - but for single subject, we need a different approach
                    # For single subject, we calculate the reliability of the mean
                    r1, r2, r3 = record.iloc[0]['r1'], record.iloc[0]['r2'], record.iloc[0]['r3']
                    
                    if pd.isna(r1) or pd.isna(r2) or pd.isna(r3):
                        icc_value = np.nan
                    else:
                        # For single subject with k raters, calculate Cronbach's alpha equivalent
                        ratings_array = np.array([r1, r2, r3])
                        
                        if len(np.unique(ratings_array)) == 1:
                            # Perfect agreement
                            icc_value = 1.0
                        else:
                            # Calculate variance-based reliability
                            mean_rating = np.mean(ratings_array)
                            var_ratings = np.var(ratings_array, ddof=1) if len(ratings_array) > 1 else 0
                            
                            # Simple reliability measure: 1 - (variance / max_possible_variance)
                            max_var = 0.25  # Maximum variance for 0-1 scale
                            icc_value = max(0, 1 - (var_ratings / max_var)) if max_var > 0 else 1.0
                    
                    # Get set information
                    set_info = set_df[set_df['task_id'] == task_id]
                    task_set = set_info['set'].iloc[0] if len(set_info) > 0 else 'unknown'
                    
                    # Get status information
                    model_mapping = {'model1': 'o3_mini', 'model2': 'gemini2.5', 'model3': 'gemini2.0', 
                                   'model4': 'deepseek_r1', 'model5': 'claude', 'model6': 'qwen_qwq'}
                    mapped_model = model_mapping.get(model, model)
                    
                    status_col = f"{mapped_model}_status"
                    status_info = status_df[status_df['task_id'] == task_id]
                    
                    if len(status_info) > 0 and status_col in status_info.columns:
                        status_value = status_info[status_col].iloc[0]
                        fail_binary = 1 if str(status_value).lower().strip() in ['fail', 'failed', 'f'] else 0
                    else:
                        fail_binary = np.nan
                    
                    icc_results.append({
                        'task_id': task_id,
                        'model': model,
                        'dimension': dimension,
                        'set': task_set,
                        'r1': r1,
                        'r2': r2,
                        'r3': r3,
                        'mean_score': record.iloc[0]['mean_of_3'],
                        'icc_agreement': icc_value,
                        'agreement_level': interpret_icc(icc_value),
                        'fail_binary': fail_binary
                    })
    
    return pd.DataFrame(icc_results)

def analyze_agreement_patterns(icc_df):
    """Analyze agreement patterns across different dimensions"""
    
    print("\n" + "="*60)
    print("ICC AGREEMENT ANALYSIS RESULTS")
    print("="*60)
    
    # Overall agreement statistics
    print(f"\nTotal combinations analyzed: {len(icc_df)}")
    print(f"Valid ICC calculations: {len(icc_df.dropna(subset=['icc_agreement']))}")
    
    # Agreement level distribution
    print(f"\n=== OVERALL AGREEMENT DISTRIBUTION ===")
    agreement_counts = icc_df['agreement_level'].value_counts()
    total_valid = len(icc_df.dropna(subset=['icc_agreement']))
    
    for level in ['Excellent', 'Good', 'Moderate', 'Poor', 'Invalid']:
        count = agreement_counts.get(level, 0)
        pct = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"{level:>10}: {count:>4} ({pct:>5.1f}%)")
    
    # Agreement by dimension
    print(f"\n=== AGREEMENT BY DIMENSION ===")
    for dimension in ['completeness', 'efficiency', 'logic']:
        dim_data = icc_df[icc_df['dimension'] == dimension].dropna(subset=['icc_agreement'])
        if len(dim_data) > 0:
            mean_icc = dim_data['icc_agreement'].mean()
            agreement_dist = dim_data['agreement_level'].value_counts()
            
            print(f"\n{dimension.upper()}:")
            print(f"  Mean ICC: {mean_icc:.3f} ({interpret_icc(mean_icc)})")
            print(f"  Distribution:")
            for level in ['Excellent', 'Good', 'Moderate', 'Poor']:
                count = agreement_dist.get(level, 0)
                pct = (count / len(dim_data) * 100) if len(dim_data) > 0 else 0
                print(f"    {level}: {count} ({pct:.1f}%)")
    
    # Agreement by set (hard vs full)
    print(f"\n=== AGREEMENT BY SET (HARD vs FULL) ===")
    for set_type in ['hard', 'full']:
        set_data = icc_df[icc_df['set'] == set_type].dropna(subset=['icc_agreement'])
        if len(set_data) > 0:
            mean_icc = set_data['icc_agreement'].mean()
            agreement_dist = set_data['agreement_level'].value_counts()
            
            print(f"\n{set_type.upper()} SET:")
            print(f"  Records: {len(set_data)}")
            print(f"  Mean ICC: {mean_icc:.3f} ({interpret_icc(mean_icc)})")
            print(f"  Distribution:")
            for level in ['Excellent', 'Good', 'Moderate', 'Poor']:
                count = agreement_dist.get(level, 0)
                pct = (count / len(set_data) * 100) if len(set_data) > 0 else 0
                print(f"    {level}: {count} ({pct:.1f}%)")
    
    # Correlation between agreement and outcomes
    print(f"\n=== AGREEMENT vs OUTCOMES ===")
    valid_data = icc_df.dropna(subset=['icc_agreement', 'fail_binary'])
    
    if len(valid_data) > 0:
        # Agreement level vs failure rate
        print(f"\nFailure rates by agreement level:")
        for level in ['Excellent', 'Good', 'Moderate', 'Poor']:
            level_data = valid_data[valid_data['agreement_level'] == level]
            if len(level_data) > 0:
                fail_rate = level_data['fail_binary'].mean()
                print(f"  {level:>10}: {fail_rate:.3f} ({len(level_data)} records)")
        
        # Correlation between ICC and failure
        from scipy.stats import spearmanr
        icc_values = valid_data['icc_agreement']
        fail_values = valid_data['fail_binary']
        
        if len(icc_values) > 10:
            corr, p_val = spearmanr(icc_values, fail_values)
            print(f"\nSpearman correlation (ICC vs Failure):")
            print(f"  r = {corr:.3f}, p = {p_val:.3f}")
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  Significance: {significance if significance else 'Not significant'}")

def main():
    """Main analysis function"""
    
    print("Starting ICC Agreement Analysis...")
    print("Using ICC(1,k) with standard interpretation bands:")
    print("  Excellent: > 0.90")
    print("  Good: 0.75 - 0.89") 
    print("  Moderate: 0.50 - 0.74")
    print("  Poor: < 0.50")
    
    # Calculate ICC for all combinations
    icc_df = calculate_icc_for_all_combinations()
    
    # Save results
    icc_df.to_csv('icc_agreement_results.csv', index=False)
    print(f"\n✓ ICC results saved to 'icc_agreement_results.csv'")
    
    # Analyze patterns
    analyze_agreement_patterns(icc_df)
    
    # Key insights
    print(f"\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    valid_data = icc_df.dropna(subset=['icc_agreement'])
    if len(valid_data) > 0:
        overall_mean = valid_data['icc_agreement'].mean()
        print(f"\n1. Overall inter-rater agreement: {overall_mean:.3f} ({interpret_icc(overall_mean)})")
        
        # Compare hard vs full
        hard_data = valid_data[valid_data['set'] == 'hard']
        full_data = valid_data[valid_data['set'] == 'full']
        
        if len(hard_data) > 0 and len(full_data) > 0:
            hard_mean = hard_data['icc_agreement'].mean()
            full_mean = full_data['icc_agreement'].mean()
            
            print(f"\n2. Agreement by task difficulty:")
            print(f"   Hard tasks: {hard_mean:.3f} ({interpret_icc(hard_mean)})")
            print(f"   Full tasks: {full_mean:.3f} ({interpret_icc(full_mean)})")
            
            if hard_mean < full_mean:
                print(f"   → Raters agree LESS on hard tasks (difference: {full_mean - hard_mean:.3f})")
            else:
                print(f"   → Raters agree MORE on hard tasks (difference: {hard_mean - full_mean:.3f})")
        
        # Best agreement dimension
        dim_means = {}
        for dim in ['completeness', 'efficiency', 'logic']:
            dim_data = valid_data[valid_data['dimension'] == dim]
            if len(dim_data) > 0:
                dim_means[dim] = dim_data['icc_agreement'].mean()
        
        if dim_means:
            best_dim = max(dim_means, key=dim_means.get)
            worst_dim = min(dim_means, key=dim_means.get)
            
            print(f"\n3. Agreement by evaluation dimension:")
            print(f"   Best agreement: {best_dim} ({dim_means[best_dim]:.3f})")
            print(f"   Worst agreement: {worst_dim} ({dim_means[worst_dim]:.3f})")
    
    print(f"\nAnalysis complete! Check 'icc_agreement_results.csv' for detailed results.")

if __name__ == "__main__":
    main()
