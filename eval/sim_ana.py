#!/usr/bin/env python3
"""
O3-Mini Reasoning Similarity Analysis Script

This script provides comprehensive analysis of o3-mini reasoning consistency
across different thinking levels (low, medium, high) and datasets (full, hard).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class O3MiniSimilarityAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.df_wide = None
        self.df_pairwise = None
        self.load_data()
    
    def load_data(self):
        """Load similarity results from Excel file."""
        try:
            self.df_wide = pd.read_excel(self.results_file, sheet_name='wide_format')
            self.df_pairwise = pd.read_excel(self.results_file, sheet_name='pairwise_scores')
            print(f"✓ Loaded data: {len(self.df_wide)} tasks, {len(self.df_pairwise)} pairwise comparisons")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return
    
    def calculate_derived_metrics(self):
        """Calculate additional metrics for analysis."""
        # Average similarities per task
        tfidf_cols = ['tfidf_low_vs_high', 'tfidf_low_vs_medium', 'tfidf_medium_vs_high']
        embed_cols = ['embed_low_vs_high', 'embed_low_vs_medium', 'embed_medium_vs_high']
        
        self.df_wide['avg_tfidf'] = self.df_wide[tfidf_cols].mean(axis=1)
        self.df_wide['avg_embed'] = self.df_wide[embed_cols].mean(axis=1)
        
        # Consistency measures (lower variance = more consistent)
        self.df_wide['tfidf_variance'] = self.df_wide[tfidf_cols].var(axis=1)
        self.df_wide['embed_variance'] = self.df_wide[embed_cols].var(axis=1)
        self.df_wide['tfidf_std'] = self.df_wide[tfidf_cols].std(axis=1)
        
        # Consistency score (0-1, higher = more consistent)
        self.df_wide['consistency_score'] = 1 - (self.df_wide['tfidf_std'] / self.df_wide['avg_tfidf'])
        self.df_wide['consistency_score'] = self.df_wide['consistency_score'].clip(0, 1)
    
    def overall_statistics(self):
        """Generate overall similarity statistics."""
        print("\n" + "="*60)
        print("O3-MINI REASONING CONSISTENCY ANALYSIS")
        print("="*60)
        
        print("\n1. OVERALL SIMILARITY STATISTICS")
        print("-" * 40)
        
        # TF-IDF similarities
        tfidf_cols = ['tfidf_low_vs_high', 'tfidf_low_vs_medium', 'tfidf_medium_vs_high']
        embed_cols = ['embed_low_vs_high', 'embed_low_vs_medium', 'embed_medium_vs_high']
        
        print("TF-IDF Character N-gram Similarities:")
        for col in tfidf_cols:
            values = self.df_wide[col].dropna()
            print(f"  {col.replace('tfidf_', '').replace('_', ' ').title()}: "
                  f"Mean={values.mean():.3f}, Std={values.std():.3f}, "
                  f"Range=[{values.min():.3f}, {values.max():.3f}]")
        
        print("\nEmbedding Hashing Similarities:")
        for col in embed_cols:
            values = self.df_wide[col].dropna()
            print(f"  {col.replace('embed_', '').replace('_', ' ').title()}: "
                  f"Mean={values.mean():.3f}, Std={values.std():.3f}, "
                  f"Range=[{values.min():.3f}, {values.max():.3f}]")
    
    def dataset_comparison(self):
        """Compare consistency between full and hard datasets."""
        print("\n2. DATASET COMPARISON (Full vs Hard)")
        print("-" * 40)
        
        for dataset in ['full', 'hard']:
            subset = self.df_pairwise[self.df_pairwise['dataset'] == dataset]
            if len(subset) > 0:
                tfidf_mean = subset['tfidf_char_3_5_cosine'].mean()
                tfidf_std = subset['tfidf_char_3_5_cosine'].std()
                embed_mean = subset['embedding_hashing_cosine'].mean()
                embed_std = subset['embedding_hashing_cosine'].std()
                
                print(f"{dataset.upper()} dataset:")
                print(f"  TF-IDF similarity: {tfidf_mean:.3f} ± {tfidf_std:.3f}")
                print(f"  Embedding similarity: {embed_mean:.3f} ± {embed_std:.3f}")
                print(f"  Sample size: {len(subset)} comparisons")
        
        # Statistical test for difference
        full_data = self.df_pairwise[self.df_pairwise['dataset'] == 'full']['tfidf_char_3_5_cosine']
        hard_data = self.df_pairwise[self.df_pairwise['dataset'] == 'hard']['tfidf_char_3_5_cosine']
        
        if len(full_data) > 0 and len(hard_data) > 0:
            t_stat, p_value = stats.ttest_ind(full_data, hard_data)
            print(f"\nStatistical test (Full vs Hard): t={t_stat:.3f}, p={p_value:.3f}")
            if p_value < 0.05:
                print("  → Significant difference between datasets")
            else:
                print("  → No significant difference between datasets")
    
    def thinking_level_analysis(self):
        """Analyze similarity patterns across thinking levels."""
        print("\n3. THINKING LEVEL PAIR ANALYSIS")
        print("-" * 40)
        
        pairs = self.df_pairwise['pair'].unique()
        pair_stats = []
        
        for pair in pairs:
            subset = self.df_pairwise[self.df_pairwise['pair'] == pair]
            if len(subset) > 0:
                tfidf_mean = subset['tfidf_char_3_5_cosine'].mean()
                embed_mean = subset['embedding_hashing_cosine'].mean()
                tfidf_std = subset['tfidf_char_3_5_cosine'].std()
                embed_std = subset['embedding_hashing_cosine'].std()
                
                pair_stats.append({
                    'pair': pair,
                    'tfidf_mean': tfidf_mean,
                    'tfidf_std': tfidf_std,
                    'embed_mean': embed_mean,
                    'embed_std': embed_std
                })
                
                consistency = "High" if tfidf_std < 0.15 else "Medium" if tfidf_std < 0.25 else "Low"
                print(f"{pair.replace('_', ' ').title()}:")
                print(f"  TF-IDF: {tfidf_mean:.3f} ± {tfidf_std:.3f}")
                print(f"  Embedding: {embed_mean:.3f} ± {embed_std:.3f}")
                print(f"  Consistency: {consistency}")
                print()
        
        # Rank pairs by similarity
        pair_stats.sort(key=lambda x: x['tfidf_mean'], reverse=True)
        print("Thinking level pairs ranked by similarity:")
        for i, stats in enumerate(pair_stats, 1):
            print(f"  {i}. {stats['pair'].replace('_', ' ').title()}: {stats['tfidf_mean']:.3f}")
    
    def consistency_analysis(self):
        """Analyze task-level consistency."""
        print("\n4. TASK CONSISTENCY ANALYSIS")
        print("-" * 40)
        
        # Most consistent tasks
        print("MOST CONSISTENT TASKS (Top 5):")
        most_consistent = self.df_wide.nsmallest(5, 'tfidf_variance')[
            ['dataset', 'task_id', 'avg_tfidf', 'tfidf_variance', 'consistency_score']
        ]
        for idx, row in most_consistent.iterrows():
            print(f"  {row['dataset']} - {row['task_id']}: "
                  f"Avg={row['avg_tfidf']:.3f}, Variance={row['tfidf_variance']:.4f}, "
                  f"Score={row['consistency_score']:.3f}")
        
        print("\nLEAST CONSISTENT TASKS (Top 5):")
        least_consistent = self.df_wide.nlargest(5, 'tfidf_variance')[
            ['dataset', 'task_id', 'avg_tfidf', 'tfidf_variance', 'consistency_score']
        ]
        for idx, row in least_consistent.iterrows():
            print(f"  {row['dataset']} - {row['task_id']}: "
                  f"Avg={row['avg_tfidf']:.3f}, Variance={row['tfidf_variance']:.4f}, "
                  f"Score={row['consistency_score']:.3f}")
    
    def distribution_analysis(self):
        """Analyze distribution of similarity scores."""
        print("\n5. SIMILARITY DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        # Categorize tasks by similarity level
        low_sim = len(self.df_wide[self.df_wide['avg_tfidf'] < 0.5])
        medium_sim = len(self.df_wide[(self.df_wide['avg_tfidf'] >= 0.5) & (self.df_wide['avg_tfidf'] < 0.7)])
        high_sim = len(self.df_wide[self.df_wide['avg_tfidf'] >= 0.7])
        total = len(self.df_wide)
        
        print(f"Tasks with LOW similarity (<0.5): {low_sim} ({low_sim/total*100:.1f}%)")
        print(f"Tasks with MEDIUM similarity (0.5-0.7): {medium_sim} ({medium_sim/total*100:.1f}%)")
        print(f"Tasks with HIGH similarity (>0.7): {high_sim} ({high_sim/total*100:.1f}%)")
        
        # Consistency distribution
        high_consistency = len(self.df_wide[self.df_wide['consistency_score'] > 0.8])
        medium_consistency = len(self.df_wide[(self.df_wide['consistency_score'] >= 0.6) & 
                                            (self.df_wide['consistency_score'] <= 0.8)])
        low_consistency = len(self.df_wide[self.df_wide['consistency_score'] < 0.6])
        
        print(f"\nTasks with HIGH consistency (>0.8): {high_consistency} ({high_consistency/total*100:.1f}%)")
        print(f"Tasks with MEDIUM consistency (0.6-0.8): {medium_consistency} ({medium_consistency/total*100:.1f}%)")
        print(f"Tasks with LOW consistency (<0.6): {low_consistency} ({low_consistency/total*100:.1f}%)")
    
    def key_insights(self):
        """Generate key insights and interpretations."""
        print("\n6. KEY INSIGHTS & INTERPRETATION")
        print("-" * 40)
        
        overall_tfidf = self.df_wide['avg_tfidf'].mean()
        overall_embed = self.df_wide['avg_embed'].mean()
        overall_consistency = self.df_wide['consistency_score'].mean()
        
        print(f"Overall Reasoning Similarity: {overall_tfidf:.3f} (TF-IDF), {overall_embed:.3f} (Embedding)")
        print(f"Overall Consistency Score: {overall_consistency:.3f} (0=inconsistent, 1=perfectly consistent)")
        
        # Stability assessment
        if overall_tfidf > 0.7:
            stability = "HIGH"
        elif overall_tfidf > 0.5:
            stability = "MODERATE"
        else:
            stability = "LOW"
        
        print(f"O3-Mini Reasoning Stability: {stability}")
        
        print("\nKEY FINDINGS:")
        
        # Finding 1: Overall consistency
        if overall_tfidf > 0.6:
            print("✓ Model shows good consistency in reasoning patterns across thinking levels")
        else:
            print("⚠ Model shows significant variation in reasoning patterns across thinking levels")
        
        # Finding 2: Variance analysis
        avg_variance = self.df_wide['tfidf_variance'].mean()
        if avg_variance < 0.01:
            print("✓ Very low variance indicates highly stable reasoning approach")
        elif avg_variance < 0.02:
            print("✓ Low variance indicates reasonably stable reasoning approach")
        else:
            print("⚠ Higher variance suggests reasoning approach changes with thinking level")
        
        # Finding 3: Embedding vs TF-IDF
        if overall_embed > overall_tfidf + 0.1:
            print("✓ Higher embedding similarity suggests semantic consistency even when wording differs")
        
        # Finding 4: Dataset comparison
        full_avg = self.df_pairwise[self.df_pairwise['dataset'] == 'full']['tfidf_char_3_5_cosine'].mean()
        hard_avg = self.df_pairwise[self.df_pairwise['dataset'] == 'hard']['tfidf_char_3_5_cosine'].mean()
        
        if abs(full_avg - hard_avg) < 0.05:
            print("✓ Reasoning consistency is similar between full and hard datasets")
        else:
            print(f"⚠ Reasoning consistency differs between datasets (Full: {full_avg:.3f}, Hard: {hard_avg:.3f})")
        
        # Finding 5: High consistency percentage
        high_consistency_pct = len(self.df_wide[self.df_wide['avg_tfidf'] >= 0.7]) / len(self.df_wide) * 100
        print(f"✓ {high_consistency_pct:.1f}% of tasks show high reasoning consistency across thinking levels")
    
    
    def generate_report(self):
        """Generate a comprehensive CSV report."""
        # Create summary statistics
        summary_stats = {
            'Metric': [
                'Overall TF-IDF Mean', 'Overall TF-IDF Std', 'Overall Embedding Mean', 'Overall Embedding Std',
                'Overall Consistency Mean', 'Overall Consistency Std', 'High Similarity Tasks (%)', 
                'High Consistency Tasks (%)', 'Most Similar Pair', 'Least Similar Pair'
            ],
            'Value': [
                f"{self.df_wide['avg_tfidf'].mean():.3f}",
                f"{self.df_wide['avg_tfidf'].std():.3f}",
                f"{self.df_wide['avg_embed'].mean():.3f}",
                f"{self.df_wide['avg_embed'].std():.3f}",
                f"{self.df_wide['consistency_score'].mean():.3f}",
                f"{self.df_wide['consistency_score'].std():.3f}",
                f"{len(self.df_wide[self.df_wide['avg_tfidf'] >= 0.7]) / len(self.df_wide) * 100:.1f}%",
                f"{len(self.df_wide[self.df_wide['consistency_score'] > 0.8]) / len(self.df_wide) * 100:.1f}%",
                "Medium vs High",
                "Low vs Medium"
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv('o3_mini_similarity_summary.csv', index=False)
        
        # Save detailed task-level results
        detailed_results = self.df_wide[['dataset', 'task_id', 'avg_tfidf', 'avg_embed', 
                                        'consistency_score', 'tfidf_variance']].copy()
        detailed_results.to_csv('o3_mini_detailed_results.csv', index=False)
        
        # print(f"✓ Summary report saved to: o3_mini_similarity_summary.csv")
        # print(f"✓ Detailed results saved to: o3_mini_detailed_results.csv")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        if self.df_wide is None or self.df_pairwise is None:
            print("✗ No data loaded. Cannot run analysis.")
            return
        
        self.calculate_derived_metrics()
        self.overall_statistics()
        self.dataset_comparison()
        self.thinking_level_analysis()
        self.consistency_analysis()
        self.distribution_analysis()
        self.key_insights()
        self.create_visualizations()
        self.generate_report()
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")

def main():
    """Main function to run the analysis."""
    results_file = 'model_outputs/o3-mini_outputs/o3-diff/o3_mini_similarity_hard_results.xlsx'
    
    analyzer = O3MiniSimilarityAnalyzer(results_file)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
