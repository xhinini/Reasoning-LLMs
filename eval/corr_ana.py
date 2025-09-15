import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# File paths
SET_PATH = "set.xlsx"
SCORES_PATH = "score.xlsx" 
STATUS_PATH = "status.xlsx"
OUT_DIR = Path(".")

def clean_task_id(task_id):
    """Clean task IDs thoroughly to ensure proper matching"""
    if pd.isna(task_id):
        return task_id
    # Strip whitespace, normalize unicode, remove any hidden characters
    cleaned = str(task_id).strip()
    # Remove any trailing newlines or carriage returns
    cleaned = cleaned.replace('\n', '').replace('\r', '').replace('\t', '')
    return cleaned

def load_and_merge_data():
    """Load and merge all three datasets with proper task ID cleaning"""
    print("Loading datasets...")
    
    # Load set information
    set_df = pd.read_excel(SET_PATH)
    print(f"Set data: {set_df.shape[0]} tasks, sets: {set_df['set'].unique()}")
    
    # Load scores
    scores_df = pd.read_excel(SCORES_PATH)
    print(f"Scores data: {scores_df.shape[0]} records")
    
    # Load status
    status_df = pd.read_excel(STATUS_PATH)
    print(f"Status data: {status_df.shape[0]} tasks")
    
    # Clean task IDs thoroughly
    set_df['task_id'] = set_df['task_id'].apply(clean_task_id)
    status_df['task_id'] = status_df['task_id'].apply(clean_task_id)
    scores_df['task_id'] = scores_df['task_id'].apply(clean_task_id)
    
    # Create model mapping
    status_cols = [c for c in status_df.columns if c.endswith('_status')]
    model_names = [c.replace('_status', '') for c in status_cols]
    score_models = sorted(scores_df['model'].unique())
    
    model_mapping = dict(zip(score_models, model_names))
    print(f"Model mapping: {model_mapping}")
    
    # Pivot scores to get metrics as columns
    scores_pivot = scores_df.pivot_table(
        index=['task_id', 'model'],
        columns='dimension',
        values='mean_of_3',
        aggfunc='mean'
    ).reset_index()
    
    print(f"After pivoting scores: {len(scores_pivot)} records from {len(scores_pivot['task_id'].unique())} unique tasks")
    
    # Map model names in scores to match status
    scores_pivot['model'] = scores_pivot['model'].map(model_mapping)
    print(f"After model mapping: {len(scores_pivot)} records")
    
    # Melt status to long format
    status_long = status_df.melt(
        id_vars=['task_id'],
        value_vars=status_cols,
        var_name='model_status',
        value_name='status'
    )
    status_long['model'] = status_long['model_status'].str.replace('_status', '')
    
    # Convert status to binary (1=fail, 0=pass)
    def status_to_binary(status):
        if pd.isna(status):
            return np.nan
        status_str = str(status).lower().strip()
        if status_str in ['fail', 'failed', 'f', 'no', '0', 'false']:
            return 1
        elif status_str in ['pass', 'passed', 'p', 'yes', '1', 'true', 'success']:
            return 0
        else:
            return np.nan
    
    status_long['fail_binary'] = status_long['status'].apply(status_to_binary)
    
    print(f"Status long format: {len(status_long)} records from {len(status_long['task_id'].unique())} unique tasks")
    
    # Debug: Check combinations before merge
    status_combos = set(zip(status_long['task_id'], status_long['model']))
    scores_combos = set(zip(scores_pivot['task_id'], scores_pivot['model']))
    
    print(f"Status combinations: {len(status_combos)}")
    print(f"Scores combinations: {len(scores_combos)}")
    
    # Check for task mismatches
    set_tasks = set(set_df['task_id'])
    score_tasks = set(scores_pivot['task_id'])
    status_tasks = set(status_long['task_id'])
    
    print(f"Tasks in set.xlsx: {len(set_tasks)}")
    print(f"Tasks in scores: {len(score_tasks)}")
    print(f"Tasks in status: {len(status_tasks)}")
    
    missing_in_set = score_tasks - set_tasks
    if missing_in_set:
        print(f"Tasks in scores but missing from set: {len(missing_in_set)}")
        for task in list(missing_in_set)[:3]:
            print(f"  '{task}' (len={len(task)})")
    
    # Merge all datasets
    merged = scores_pivot.merge(set_df, on='task_id', how='inner')
    print(f"After merging with set data: {merged.shape[0]} records")
    
    if len(merged) < len(scores_pivot):
        print(f"WARNING: Lost {len(scores_pivot) - len(merged)} records in set merge!")
    
    # Check set distribution after first merge
    if len(merged) > 0:
        print(f"Set distribution after first merge: {merged['set'].value_counts().to_dict()}")
    
    merged = merged.merge(
        status_long[['task_id', 'model', 'fail_binary']], 
        on=['task_id', 'model'], 
        how='inner'
    )
    print(f"After merging with status data: {merged.shape[0]} records")
    
    # Remove rows with missing status
    merged = merged.dropna(subset=['fail_binary'])
    print(f"After removing missing status: {merged.shape[0]} records")
    
    # Create composite metrics
    metric_cols = [c for c in merged.columns if c in ['completeness', 'logic', 'efficiency']]
    print(f"Available metric columns: {metric_cols}")
    
    if len(metric_cols) >= 2:
        merged['avg_score'] = merged[metric_cols].mean(axis=1)
        merged['min_score'] = merged[metric_cols].min(axis=1)
        merged['max_score'] = merged[metric_cols].max(axis=1)
    
    print(f"Final merged dataset: {merged.shape[0]} records")
    if len(merged) > 0:
        print(f"Sets distribution: {merged['set'].value_counts().to_dict()}")
        print(f"Sample of merged data:")
        print(merged[['task_id', 'model', 'set', 'fail_binary'] + metric_cols].head())
    
    return merged, metric_cols

def calculate_correlations(data, metrics, set_name):
    """Calculate correlations for a specific set"""
    results = []
    
    for metric in metrics:
        if metric not in data.columns:
            continue
            
        # Remove missing values
        clean_data = data[[metric, 'fail_binary']].dropna()
        
        if len(clean_data) < 10:  
            continue
            
        x = clean_data[metric]
        y = clean_data['fail_binary']
        
        # Calculate different correlation measures
        try:
            spearman_r, spearman_p = spearmanr(x, y)
        except:
            spearman_r, spearman_p = np.nan, np.nan
            
        try:
            pearson_r, pearson_p = pearsonr(x, y)
        except:
            pearson_r, pearson_p = np.nan, np.nan
            
        try:
            kendall_tau, kendall_p = kendalltau(x, y)
        except:
            kendall_tau, kendall_p = np.nan, np.nan
        
        # Calculate failure rates by metric quartiles
        quartiles = np.percentile(x, [25, 50, 75])
        q1_fail_rate = y[x <= quartiles[0]].mean() if len(y[x <= quartiles[0]]) > 0 else np.nan
        q4_fail_rate = y[x >= quartiles[2]].mean() if len(y[x >= quartiles[2]]) > 0 else np.nan
        
        results.append({
            'set': set_name,
            'metric': metric,
            'n_samples': len(clean_data),
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p,
            'q1_fail_rate': q1_fail_rate,
            'q4_fail_rate': q4_fail_rate,
            'q1_q4_diff': q1_fail_rate - q4_fail_rate if not pd.isna(q1_fail_rate) and not pd.isna(q4_fail_rate) else np.nan
        })
    
    return results

def main():
    """Main analysis function"""
    
    print("Starting FIXED correlation analysis by set...")
    
    merged_data, metric_cols = load_and_merge_data()
    
    if len(merged_data) == 0:
        print("ERROR: No data to analyze after merging!")
        return
    
    # Perform analysis by set
    all_individual_results = []
    
    for set_type in ['hard', 'full']:
        set_data = merged_data[merged_data['set'] == set_type].copy()
        
        print(f"\n=== Analyzing {set_type.upper()} set ===")
        print(f"Sample size: {len(set_data)} records")
        print(f"Failure rate: {set_data['fail_binary'].mean():.3f}")
        
        # Individual metrics
        individual_results = calculate_correlations(set_data, metric_cols + ['avg_score', 'min_score', 'max_score'], set_type)
        all_individual_results.extend(individual_results)
    
    print("\nSaving results...")
    
    pd.DataFrame(all_individual_results).to_csv(OUT_DIR / "correlation_results_fixed.csv", index=False)
    print("✓ Fixed correlation results saved")
    
    print("\n" + "="*60)
    print("FIXED ANALYSIS COMPLETE - KEY RESULTS")
    print("="*60)
    
    print("\nIndividual Metric Correlations (Top 5 by absolute correlation):")
    df_individual = pd.DataFrame(all_individual_results)
    df_individual['abs_correlation'] = abs(df_individual['spearman_r'])
    top_correlations = df_individual.nlargest(5, 'abs_correlation')
    
    for _, row in top_correlations.iterrows():
        significance = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else ""
        print(f"  {row['set'].upper()} - {row['metric']}: r={row['spearman_r']:.3f} (p={row['spearman_p']:.3f}){significance}")
    
    expected_records = 600 - 8  
    actual_records = len(merged_data)
    print(f"\nRecord count verification:")
    print(f"Expected records (600 - 8 missing status): {expected_records}")
    print(f"Actual records: {actual_records}")
    print(f"Match: {'✓' if expected_records == actual_records else '✗'}")

if __name__ == "__main__":
    main()
