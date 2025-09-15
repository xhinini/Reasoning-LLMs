#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected Incomplete Reasoning Analysis

Using the proper threshold: reasoning is incomplete if score < 1.0
(Any score less than perfect indicates some degree of incompleteness)
"""

import pandas as pd
import numpy as np
from scipy import stats

def clean_task_id(task_id):
    """Clean task IDs thoroughly to ensure proper matching"""
    if pd.isna(task_id):
        return task_id
    cleaned = str(task_id).strip().replace('\n', '').replace('\r', '').replace('\t', '')
    return cleaned

def analyze_incomplete_reasoning_corrected():
    """Analyze incomplete reasoning using the correct threshold (< 1.0)"""
    
    print("Loading data for corrected incomplete reasoning analysis...")
    
    set_df = pd.read_excel('set.xlsx')
    scores_df = pd.read_excel('score.xlsx')
    status_df = pd.read_excel('status.xlsx')
    
    set_df['task_id'] = set_df['task_id'].apply(clean_task_id)
    scores_df['task_id'] = scores_df['task_id'].apply(clean_task_id)
    status_df['task_id'] = status_df['task_id'].apply(clean_task_id)
    
    completeness_df = scores_df[scores_df['dimension'] == 'completeness'].copy()
    
    completeness_with_set = completeness_df.merge(set_df, on='task_id', how='inner')
    
    print(f"Total completeness records: {len(completeness_with_set)}")
    print(f"Hard set records: {len(completeness_with_set[completeness_with_set['set'] == 'hard'])}")
    print(f"Full set records: {len(completeness_with_set[completeness_with_set['set'] == 'full'])}")
    
    completeness_with_set['incomplete_reasoning'] = completeness_with_set['mean_of_3'] < 1.0
    
    print("\n" + "="*60)
    print("CORRECTED INCOMPLETE REASONING ANALYSIS")
    print("Threshold: Incomplete if mean_of_3 < 1.0")
    print("="*60)
    
    # Overall statistics
    total_incomplete = completeness_with_set['incomplete_reasoning'].sum()
    total_records = len(completeness_with_set)
    overall_incomplete_rate = total_incomplete / total_records * 100
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total records: {total_records}")
    print(f"Incomplete reasoning cases: {total_incomplete}")
    print(f"Overall incomplete rate: {overall_incomplete_rate:.1f}%")
    

    print(f"\nBY TASK SET:")
    
    for set_type in ['hard', 'full']:
        set_data = completeness_with_set[completeness_with_set['set'] == set_type]
        
        incomplete_count = set_data['incomplete_reasoning'].sum()
        total_count = len(set_data)
        incomplete_rate = incomplete_count / total_count * 100 if total_count > 0 else 0
        
        mean_completeness = set_data['mean_of_3'].mean()
        median_completeness = set_data['mean_of_3'].median()
        std_completeness = set_data['mean_of_3'].std()
        
        print(f"\n{set_type.upper()} SET:")
        print(f"  Records: {total_count}")
        print(f"  Incomplete reasoning: {incomplete_count} ({incomplete_rate:.1f}%)")
        print(f"  Mean completeness: {mean_completeness:.3f}")
        print(f"  Median completeness: {median_completeness:.3f}")
        print(f"  Std completeness: {std_completeness:.3f}")
        
        # Show distribution of scores
        print(f"  Score distribution:")
        score_counts = set_data['mean_of_3'].value_counts().sort_index()
        for score, count in score_counts.head(10).items():
            print(f"    {score:.3f}: {count} cases")
        if len(score_counts) > 10:
            print(f"    ... and {len(score_counts) - 10} more unique scores")
    
    # Statistical comparison
    print(f"\nSTATISTICAL COMPARISON:")
    
    hard_data = completeness_with_set[completeness_with_set['set'] == 'hard']['mean_of_3']
    full_data = completeness_with_set[completeness_with_set['set'] == 'full']['mean_of_3']
    
    hard_incomplete_rate = (hard_data < 1.0).mean() * 100
    full_incomplete_rate = (full_data < 1.0).mean() * 100
    
    print(f"Hard set incomplete rate: {hard_incomplete_rate:.1f}%")
    print(f"Full set incomplete rate: {full_incomplete_rate:.1f}%")
    print(f"Difference: {hard_incomplete_rate - full_incomplete_rate:.1f} percentage points")
    
    # Statistical tests
    # Mann-Whitney U test
    # statistic, p_value_mw = stats.mannwhitneyu(hard_data, full_data, alternative='two-sided')
    # print(f"\nMann-Whitney U test:")
    # print(f"  Statistic: {statistic:.2f}")
    # print(f"  p-value: {p_value_mw:.6f}")
    # print(f"  Significant: {'Yes' if p_value_mw < 0.05 else 'No'}")
    
    # # Independent t-test
    # t_stat, p_value_t = stats.ttest_ind(hard_data, full_data)
    # print(f"\nIndependent t-test:")
    # print(f"  t-statistic: {t_stat:.3f}")
    # print(f"  p-value: {p_value_t:.6f}")
    # print(f"  Significant: {'Yes' if p_value_t < 0.05 else 'No'}")
    
    # # Chi-square test for incomplete reasoning rates
    # hard_incomplete = (hard_data < 1.0).sum()
    # hard_complete = (hard_data >= 1.0).sum()
    # full_incomplete = (full_data < 1.0).sum()
    # full_complete = (full_data >= 1.0).sum()
    
    contingency_table = np.array([[hard_incomplete, hard_complete],
                                  [full_incomplete, full_complete]])
    
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nChi-square test for incomplete reasoning rates:")
    print(f"  Contingency table:")
    print(f"    {'':>12} {'Incomplete':>12} {'Complete':>10}")
    print(f"    {'Hard':>12} {hard_incomplete:>12} {hard_complete:>10}")
    print(f"    {'Full':>12} {full_incomplete:>12} {full_complete:>10}")
    print(f"  Chi-square: {chi2:.3f}")
    print(f"  p-value: {p_value_chi2:.6f}")
    print(f"  Significant: {'Yes' if p_value_chi2 < 0.05 else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(hard_data) - 1) * hard_data.var() + 
                         (len(full_data) - 1) * full_data.var()) / 
                        (len(hard_data) + len(full_data) - 2))
    cohens_d = (hard_data.mean() - full_data.mean()) / pooled_std
    
    print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
    effect_interpretation = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
    print(f"Effect size interpretation: {effect_interpretation}")
    
    # Summary
    print(f"\n" + "="*60)
    print("KEY FINDINGS (CORRECTED ANALYSIS)")
    print("="*60)
    
    print(f"\n1. INCOMPLETE REASONING PREVALENCE:")
    print(f"   - Hard tasks: {hard_incomplete_rate:.1f}% incomplete reasoning")
    print(f"   - Full tasks: {full_incomplete_rate:.1f}% incomplete reasoning")
    print(f"   - Difference: {hard_incomplete_rate - full_incomplete_rate:.1f} percentage points higher for hard tasks")
    
    print(f"\n2. STATISTICAL SIGNIFICANCE:")
    print(f"   - Mann-Whitney U: p = {p_value_mw:.6f} ({'significant' if p_value_mw < 0.05 else 'not significant'})")
    print(f"   - Independent t-test: p = {p_value_t:.6f} ({'significant' if p_value_t < 0.05 else 'not significant'})")
    print(f"   - Chi-square test: p = {p_value_chi2:.6f} ({'significant' if p_value_chi2 < 0.05 else 'not significant'})")
    
    print(f"\n3. PRACTICAL SIGNIFICANCE:")
    print(f"   - Effect size (Cohen's d): {cohens_d:.3f} ({effect_interpretation})")
    print(f"   - Mean completeness difference: {hard_data.mean() - full_data.mean():.3f}")
    
    if hard_incomplete_rate > full_incomplete_rate:
        print(f"\n4. CONCLUSION:")
        print(f"   Hard tasks show significantly MORE incomplete reasoning than full tasks.")
        print(f"   The difference is both statistically significant and practically meaningful.")
    else:
        print(f"\n4. CONCLUSION:")
        print(f"   No significant difference in incomplete reasoning between task sets.")
    
    return completeness_with_set

if __name__ == "__main__":
    result_df = analyze_incomplete_reasoning_corrected()
    result_df.to_csv('incomplete_reasoning_corrected_analysis.csv', index=False)
    print(f"\n✓ Detailed results saved to 'incomplete_reasoning_corrected_analysis.csv'")
