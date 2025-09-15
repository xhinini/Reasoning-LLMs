#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Analysis by Set (Hard vs Full)

This script analyzes the correlation between metrics and status for both 'hard' and 'full' sets
to determine if better metrics correspond to better results in each set type.

Outputs:
- correlation_results_by_set.csv: Individual metric correlations for each set
- correlation_results_combinations.csv: Metric combination correlations for each set
- summary_comparison.csv: Summary comparing hard vs full set correlations
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SET_PATH = "set.xlsx"
SCORES_PATH = "score.xlsx" 
STATUS_PATH = "status.xlsx"
OUT_DIR = Path(".")

def load_and_merge_data():
    """Load and merge all three datasets"""
    print("Loading datasets...")
    
    set_df = pd.read_excel(SET_PATH)
    print(f"Set data: {set_df.shape[0]} tasks, sets: {set_df['set'].unique()}")
    
    scores_df = pd.read_excel(SCORES_PATH)
    print(f"Scores data: {scores_df.shape[0]} records")
    print(f"Score models: {scores_df['model'].unique()}")
    
    status_df = pd.read_excel(STATUS_PATH)
    print(f"Status data: {status_df.shape[0]} tasks")


    status_cols = [c for c in status_df.columns if c.endswith('_status')]
    model_names = [c.replace('_status', '') for c in status_cols]
    score_models = sorted(scores_df['model'].unique())
    
    print(f"Status models: {model_names}")
    print(f"Score models: {score_models}")
    
    if len(model_names) == len(score_models):
        model_mapping = dict(zip(score_models, model_names))
        print(f"Model mapping: {model_mapping}")
    else:
        print("Warning: Number of models don't match, using direct mapping")
        model_mapping = {}
    
    scores_pivot = scores_df.pivot_table(
        index=['task_id', 'model'],
        columns='dimension',
        values='mean_of_3',
        aggfunc='mean'
    ).reset_index()
    
    print(f"After pivoting scores: {len(scores_pivot)} records from {len(scores_pivot['task_id'].unique())} unique tasks")
    
    if model_mapping:
        scores_pivot['model'] = scores_pivot['model'].map(model_mapping)
        print(f"After model mapping: {len(scores_pivot)} records")
    
    status_long = status_df.melt(
        id_vars=['task_id'],
        value_vars=status_cols,
        var_name='model_status',
        value_name='status'
    )
    status_long['model'] = status_long['model_status'].str.replace('_status', '')
    
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
    
    def clean_task_id(task_id):
        if pd.isna(task_id):
            return task_id
        cleaned = str(task_id).strip()
        cleaned = cleaned.replace('\n', '').replace('\r', '')
        return cleaned
    
    set_df['task_id'] = set_df['task_id'].apply(clean_task_id)
    status_df['task_id'] = status_df['task_id'].apply(clean_task_id)
    scores_df['task_id'] = scores_df['task_id'].apply(clean_task_id)
    
    print(f"Status long format: {len(status_long)} records from {len(status_long['task_id'].unique())} unique tasks")
    print(f"Scores after mapping: {scores_pivot['model'].unique()}")
    print(f"Status models: {status_long['model'].unique()}")
    
    status_combos = set(zip(status_long['task_id'], status_long['model']))
    scores_combos = set(zip(scores_pivot['task_id'], scores_pivot['model']))
    
    print(f"Status combinations: {len(status_combos)}")
    print(f"Scores combinations: {len(scores_combos)}")
    
    missing_in_scores = status_combos - scores_combos
    missing_in_status = scores_combos - status_combos
    
    if missing_in_scores:
        print(f"Combinations in status but missing from scores: {len(missing_in_scores)}")
        for task, model in list(missing_in_scores)[:5]:
            print(f"  {task} - {model}")
    
    if missing_in_status:
        print(f"Combinations in scores but missing from status: {len(missing_in_status)}")
        for task, model in list(missing_in_status)[:5]:
            print(f"  {task} - {model}")
    
    merged = scores_pivot.merge(set_df, on='task_id', how='inner')
    print(f"After merging with set data: {merged.shape[0]} records")
    
    if len(merged) > 0:
        print(f"Set distribution after first merge: {merged['set'].value_counts().to_dict()}")
    
    merged = merged.merge(
        status_long[['task_id', 'model', 'fail_binary']], 
        on=['task_id', 'model'], 
        how='inner'
    )
    print(f"After merging with status data: {merged.shape[0]} records")
    
    merged = merged.dropna(subset=['fail_binary'])
    print(f"After removing missing status: {merged.shape[0]} records")
    
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
            
        clean_data = data[[metric, 'fail_binary']].dropna()
        
        if len(clean_data) < 10: 
            continue
            
        x = clean_data[metric]
        y = clean_data['fail_binary']
        
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

def calculate_combination_correlations(data, base_metrics, set_name):
    """Calculate correlations for metric combinations"""
    results = []
    
    for i in range(len(base_metrics)):
        for j in range(i+1, len(base_metrics)):
            metric1, metric2 = base_metrics[i], base_metrics[j]
            
            if metric1 not in data.columns or metric2 not in data.columns:
                continue
                
            clean_data = data[[metric1, metric2, 'fail_binary']].dropna()
            
            if len(clean_data) < 10:
                continue
                
            combo_avg = (clean_data[metric1] + clean_data[metric2]) / 2 
            combo_min = np.minimum(clean_data[metric1], clean_data[metric2])
            combo_prod = clean_data[metric1] * clean_data[metric2]
            
            y = clean_data['fail_binary']
            
            for combo_name, combo_values in [
                (f"{metric1}+{metric2}_avg", combo_avg),
                (f"{metric1}+{metric2}_min", combo_min),
                (f"{metric1}+{metric2}_prod", combo_prod)
            ]:
                try:
                    spearman_r, spearman_p = spearmanr(combo_values, y)
                    pearson_r, pearson_p = pearsonr(combo_values, y)
                    
                    results.append({
                        'set': set_name,
                        'combination': combo_name,
                        'n_samples': len(clean_data),
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p
                    })
                except:
                    continue
    
    return results

def analyze_by_set(merged_data, metric_cols):
    """Perform analysis separately for each set"""
    
    all_individual_results = []
    all_combination_results = []
    
    for set_type in ['hard', 'full']:
        set_data = merged_data[merged_data['set'] == set_type].copy()
        
        print(f"\n=== Analyzing {set_type.upper()} set ===")
        print(f"Sample size: {len(set_data)} records")
        print(f"Failure rate: {set_data['fail_binary'].mean():.3f}")
        
        # Individual metrics
        individual_results = calculate_correlations(set_data, metric_cols + ['avg_score', 'min_score', 'max_score'], set_type)
        all_individual_results.extend(individual_results)
        
        # Metric combinations
        combination_results = calculate_combination_correlations(set_data, metric_cols, set_type)
        all_combination_results.extend(combination_results)
    
    return all_individual_results, all_combination_results

def create_comparison_summary(individual_results):
    """Create summary comparing hard vs full sets"""
    
    if not individual_results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(individual_results)
    
    if df_results.empty or 'metric' not in df_results.columns:
        return pd.DataFrame()
    
    summary_data = []
    
    for metric in df_results['metric'].unique():
        hard_data = df_results[(df_results['metric'] == metric) & (df_results['set'] == 'hard')]
        full_data = df_results[(df_results['metric'] == metric) & (df_results['set'] == 'full')]
        
        if len(hard_data) > 0 and len(full_data) > 0:
            hard_row = hard_data.iloc[0]
            full_row = full_data.iloc[0]
            
            summary_data.append({
                'metric': metric,
                'hard_spearman_r': hard_row['spearman_r'],
                'hard_spearman_p': hard_row['spearman_p'],
                'hard_n': hard_row['n_samples'],
                'full_spearman_r': full_row['spearman_r'],
                'full_spearman_p': full_row['spearman_p'],
                'full_n': full_row['n_samples'],
                'correlation_diff': abs(hard_row['spearman_r']) - abs(full_row['spearman_r']) if not pd.isna(hard_row['spearman_r']) and not pd.isna(full_row['spearman_r']) else np.nan,
                'hard_q1_q4_diff': hard_row['q1_q4_diff'],
                'full_q1_q4_diff': full_row['q1_q4_diff']
            })
    
    return pd.DataFrame(summary_data)

def generate_report(merged_data, individual_results, combination_results, summary_df):
    """Generate comprehensive analysis report"""
    
    report_lines = []
    report_lines.append("CORRELATION ANALYSIS REPORT: HARD vs FULL SETS")
    report_lines.append("=" * 60)
    report_lines.append(f"\nDATASET OVERVIEW:")
    report_lines.append(f"Total records: {len(merged_data)}")
    
    for set_type in ['hard', 'full']:
        set_data = merged_data[merged_data['set'] == set_type]
        report_lines.append(f"{set_type.capitalize()} set: {len(set_data)} records, failure rate: {set_data['fail_binary'].mean():.3f}")

    report_lines.append(f"\nKEY FINDINGS:")
    
    df_individual = pd.DataFrame(individual_results)
    
    for set_type in ['hard', 'full']:
        set_results = df_individual[df_individual['set'] == set_type].copy()
        if len(set_results) > 0:
            set_results['abs_spearman'] = abs(set_results['spearman_r'])
            best_metric = set_results.loc[set_results['abs_spearman'].idxmax()]
            
            report_lines.append(f"\n{set_type.capitalize()} set - Best correlation:")
            report_lines.append(f"  Metric: {best_metric['metric']}")
            report_lines.append(f"  Spearman r: {best_metric['spearman_r']:.3f} (p={best_metric['spearman_p']:.3f})")
            report_lines.append(f"  Q1 vs Q4 failure rate difference: {best_metric['q1_q4_diff']:.3f}")
    
    report_lines.append(f"\nCOMPARISON INSIGHTS:")
    
    if len(summary_df) > 0:
        stronger_in_hard = summary_df[summary_df['correlation_diff'] > 0.05]
        if len(stronger_in_hard) > 0:
            report_lines.append(f"\nMetrics with stronger correlation in HARD set:")
            for _, row in stronger_in_hard.iterrows():
                report_lines.append(f"  {row['metric']}: Hard r={row['hard_spearman_r']:.3f}, Full r={row['full_spearman_r']:.3f}")
        
        stronger_in_full = summary_df[summary_df['correlation_diff'] < -0.05]
        if len(stronger_in_full) > 0:
            report_lines.append(f"\nMetrics with stronger correlation in FULL set:")
            for _, row in stronger_in_full.iterrows():
                report_lines.append(f"  {row['metric']}: Hard r={row['hard_spearman_r']:.3f}, Full r={row['full_spearman_r']:.3f}")
    
    if len(combination_results) > 0:
        df_combo = pd.DataFrame(combination_results)
        report_lines.append(f"\nBEST METRIC COMBINATIONS:")
        
        for set_type in ['hard', 'full']:
            set_combos = df_combo[df_combo['set'] == set_type].copy()
            if len(set_combos) > 0:
                set_combos['abs_spearman'] = abs(set_combos['spearman_r'])
                best_combo = set_combos.loc[set_combos['abs_spearman'].idxmax()]
                
                report_lines.append(f"\n{set_type.capitalize()} set best combination:")
                report_lines.append(f"  Combination: {best_combo['combination']}")
                report_lines.append(f"  Spearman r: {best_combo['spearman_r']:.3f} (p={best_combo['spearman_p']:.3f})")
    
    return "\n".join(report_lines)

def main():
    """Main analysis function"""
    
    print("Starting correlation analysis by set...")
    
    merged_data, metric_cols = load_and_merge_data()
    
    individual_results, combination_results = analyze_by_set(merged_data, metric_cols)
    
    summary_df = create_comparison_summary(individual_results)
    
    print("\nSaving results...")
    
    pd.DataFrame(individual_results).to_csv(OUT_DIR / "correlation_results_by_set.csv", index=False)
    print("✓ Individual metric correlations saved")
    
    if len(combination_results) > 0:
        pd.DataFrame(combination_results).to_csv(OUT_DIR / "correlation_results_combinations.csv", index=False)
        print("✓ Combination correlations saved")
    
    summary_df.to_csv(OUT_DIR / "summary_comparison.csv", index=False)
    print("✓ Summary comparison saved")
    # report = generate_report(merged_data, individual_results, combination_results, summary_df)
    # with open(OUT_DIR / "detailed_analysis_report.txt", "w") as f:
    #     f.write(report)
    # print("✓ Detailed report saved")
    
    # Display key results
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - KEY RESULTS")
    print("="*60)
    
    print("\nIndividual Metric Correlations (Top 5 by absolute correlation):")
    df_individual = pd.DataFrame(individual_results)
    df_individual['abs_correlation'] = abs(df_individual['spearman_r'])
    top_correlations = df_individual.nlargest(5, 'abs_correlation')
    
    for _, row in top_correlations.iterrows():
        significance = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else ""
        print(f"  {row['set'].upper()} - {row['metric']}: r={row['spearman_r']:.3f} (p={row['spearman_p']:.3f}){significance}")
    
    print(f"\nFiles saved:")
    print(f"  - correlation_results_by_set.csv")
    print(f"  - correlation_results_combinations.csv") 
    print(f"  - summary_comparison.csv")
    print(f"  - detailed_analysis_report.txt")

if __name__ == "__main__":
    main()
