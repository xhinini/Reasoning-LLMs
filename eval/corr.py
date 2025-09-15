#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce three deliverables from two Excel files:
  1) "Significant, metrics-only" results (Spearman, decile gap, logistic ORs with model FE)
  2) Threshold suggestions to call "Incomplete" vs "Complete" (percentile cut rules)
  3) Within-task concordance: does higher metric align with PASS within the same task?

Outputs (saved to the same folder as the inputs):
  - metrics_only_overall_correlations.csv
  - metrics_only_logit.csv  (if statsmodels available)
  - metrics_only_auc_thresholds.csv
  - threshold_suggestions_completeness.csv
  - within_task_concordance_summary.csv
  - within_task_examples_completeness.csv
  - within_task_examples_logic.csv
  - within_task_examples_efficiency.csv
  - failure_rate_deciles_Rmin_Ravg.png
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import spearmanr, kendalltau, pointbiserialr, binomtest

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except Exception:
    HAVE_SM = False

SCORES_PATH = "score.xlsx"
STATUS_PATH = "status.xlsx"

OUT_DIR = Path(STATUS_PATH).parent

def to_fail_indicator(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    # treat these as FAIL=1 vs PASS=0
    if s in {"fail", "failed", "0", "false", "f", "no", "x", "NO"}:
        return 1
    if s in {"pass", "passed", "1", "true", "t", "yes", "y", "ok", "success", "YES"}:
        return 0
    return np.nan

def load_status_long(status_path: str) -> pd.DataFrame:
    df = pd.read_excel(status_path)
    # status columns end with "_status"
    status_cols = [c for c in df.columns if str(c).lower().endswith("_status")]
    if "task_id" not in df.columns:
        raise RuntimeError("Could not find 'task_id' in status file.")
    long = df.melt(
        id_vars=["task_id"],
        value_vars=status_cols,
        var_name="status_col",
        value_name="status_str",
    )
    long["model"] = long["status_col"].apply(lambda c: f"model{status_cols.index(c)+1}")
    long["model_name"] = long["status_col"].str.replace("_status", "", regex=False)
    long["FAIL"] = long["status_str"].apply(to_fail_indicator)
    return long

def load_scores_pivot(scores_path: str) -> pd.DataFrame:
    df = pd.read_excel(scores_path)
    # Expect columns: task_id, model, dimension, mean_of_3 (already averaged across raters)
    for req in ["task_id", "model", "dimension"]:
        if req not in df.columns:
            raise RuntimeError(f"Scores file missing required column: {req}")
    if "mean_of_3" not in df.columns:
        raise RuntimeError("Scores file missing 'mean_of_3'.")
    df["dimension"] = df["dimension"].astype(str).str.strip().str.lower()
    df["mean_of_3"] = pd.to_numeric(df["mean_of_3"], errors="coerce")
    pivot = df.pivot_table(
        index=["task_id", "model"],
        columns="dimension",
        values="mean_of_3",
        aggfunc="mean"
    ).reset_index()
    for dim in ["completeness", "logic", "efficiency"]:
        if dim not in pivot.columns:
            pivot[dim] = np.nan
    return pivot

def merge_metrics_status(scores_pivot: pd.DataFrame, status_long: pd.DataFrame) -> pd.DataFrame:
    merged = scores_pivot.merge(
        status_long[["task_id", "model", "model_name", "FAIL"]],
        on=["task_id", "model"],
        how="inner",
    )
    merged = merged[merged["FAIL"].isin([0, 1])].copy()
    # Simple composites
    merged["R_avg"] = merged[["completeness", "logic", "efficiency"]].mean(axis=1)
    merged["R_min"] = merged[["completeness", "logic", "efficiency"]].min(axis=1)
    merged["R_median"] = merged[["completeness", "logic", "efficiency"]].median(axis=1)
    return merged

status_long = load_status_long(STATUS_PATH)
scores_pivot = load_scores_pivot(SCORES_PATH)
merged = merge_metrics_status(scores_pivot, status_long)

N_total_pairs = merged.shape[0]
N_fail = int((merged["FAIL"]==1).sum())
N_pass = int((merged["FAIL"]==0).sum())


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for a 1-D array of p-values."""
    p = np.array(pvals, dtype=float)
    mask = ~np.isnan(p)
    n = mask.sum()
    adj = np.full_like(p, np.nan, dtype=float)
    if n == 0:
        return adj
    
    valid_p = p[mask]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    adj_vals = ranked * n / (np.arange(1, n+1))

    adj_vals = np.minimum.accumulate(adj_vals[::-1])[::-1]
    
    temp_adj = np.full(n, np.nan)
    temp_adj[order] = np.clip(adj_vals, 0, 1)
    adj[mask] = temp_adj
    return adj

def corrs_one(x: pd.Series, y: pd.Series):
    m = x.notna() & y.notna()
    x = x[m]; y = y[m]
    n = int(len(x))
    if n < 3:
        return dict(spearman_r=np.nan, spearman_p=np.nan,
                    pearson_pb=np.nan, pearson_p=np.nan,
                    kendall_tau=np.nan, kendall_p=np.nan,
                    n=n)
    rs, ps = spearmanr(x, y)
    try:
        rpb, ppb = pointbiserialr(y, x)
    except Exception:
        rpb, ppb = np.nan, np.nan
    try:
        kt, pk = kendalltau(x, y)
    except Exception:
        kt, pk = np.nan, np.nan
    return dict(spearman_r=float(rs), spearman_p=float(ps),
                pearson_pb=float(rpb), pearson_p=float(ppb),
                kendall_tau=float(kt), kendall_p=float(pk),
                n=n)

metrics_main = ["completeness", "logic", "efficiency", "R_min", "R_avg", "R_median"]
rows = []
for col in metrics_main:
    rows.append({"metric": col, **corrs_one(merged[col], merged["FAIL"])})
overall_corrs = pd.DataFrame(rows)
overall_corrs["spearman_fdr"] = bh_fdr(overall_corrs["spearman_p"].values)
overall_corrs["kendall_fdr"] = bh_fdr(overall_corrs["kendall_p"].values)

overall_corrs.to_csv(OUT_DIR/"metrics_only_overall_correlations.csv", index=False)


logit_rows = []
if HAVE_SM:
    for col in ["completeness", "logic", "efficiency", "R_min", "R_avg", "R_median"]:
        df = merged[[col, "FAIL", "model_name"]].dropna().copy()
        if df[col].nunique() < 3 or df.shape[0] < 40:
            continue
        # Standardize predictor (1 SD change)
        df["z"] = (df[col] - df[col].mean()) / (df[col].std(ddof=0) + 1e-12)
        model = smf.logit("FAIL ~ z + C(model_name)", data=df).fit(disp=False)
        coef = float(model.params["z"])
        se = float(model.bse["z"])
        pval = float(model.pvalues["z"])
        OR = float(np.exp(coef))
        lo = float(np.exp(coef - 1.96*se))
        hi = float(np.exp(coef + 1.96*se))
        logit_rows.append({
            "metric": col,
            "OR_per_1SD": OR,
            "CI95_low": lo,
            "CI95_high": hi,
            "p_value": pval,
            "n": int(df.shape[0])
        })
    pd.DataFrame(logit_rows).sort_values("p_value").to_csv(
        OUT_DIR/"metrics_only_logit.csv", index=False
    )
else:
    print("[INFO] statsmodels not installed -> skipping logistic FE. Install via: pip install statsmodels")

# AUC/threshold-oriented summary
def roc_auc_rank(y_true: pd.Series, score: pd.Series) -> float:
    """AUC via Mann–Whitney U (no sklearn)."""
    y = y_true.values
    s = score.values
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan
    n1 = np.sum(y == 1); n0 = np.sum(y == 0)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    avg_r_pos = ranks[y == 1].mean()
    U = n1*avg_r_pos - n1*(n1+1)/2
    return float(U / (n0*n1))

def youden_gap(y: pd.Series, s: pd.Series):
    """Scan unique thresholds to maximize Youden's J; return (sens, spec, thr)."""
    df = pd.DataFrame({"y": y, "s": s}).dropna()
    if df["y"].nunique() < 2 or len(df) < 20:
        return np.nan, np.nan, np.nan
    vals = np.unique(df["s"])
    bestJ, best = -1, (np.nan, np.nan, np.nan)
    for t in vals:
        pred = (df["s"] >= t).astype(int)  # classify as FAIL if score >= t (assumes higher score = more FAIL)
        tp = ((pred == 1) & (df["y"] == 1)).sum()
        fn = ((pred == 0) & (df["y"] == 1)).sum()
        tn = ((pred == 0) & (df["y"] == 0)).sum()
        fp = ((pred == 1) & (df["y"] == 0)).sum()
        sens = tp/(tp+fn+1e-12)
        spec = tn/(tn+fp+1e-12)
        J = sens + spec - 1
        if J > bestJ:
            bestJ, best = J, (sens, spec, t)
    return best

summary_rows = []
for name, direction in {"completeness": "higher_is_better", "R_min": "higher_is_better", "R_avg": "higher_is_better"}.items():
    # For FAIL, "risk" is lower metric; AUC on (1 - metric) gives FAIL-ranking AUC
    auc = roc_auc_rank(merged["FAIL"], 1 - merged[name])
    # Youden on metric as-is: compute fail-rate gaps by bottom vs top quartiles too
    s = merged[name]; y = merged["FAIL"]
    lo = np.nanpercentile(s, 25); hi = np.nanpercentile(s, 75)
    bottom = y[s <= lo].mean(); top = y[s >= hi].mean()
    summary_rows.append({
        "metric": name,
        "ROC_AUC_for_FAIL_using_1_minus_metric": auc,
        "fail_rate_bottom25%": bottom,
        "fail_rate_top25%": top,
        "diff_pp": bottom - top
    })
pd.DataFrame(summary_rows).to_csv(OUT_DIR/"metrics_only_auc_thresholds.csv", index=False)

# Decile failure rate plot (optional visualization)
try:
    import matplotlib.pyplot as plt
    def decile_curve(series, y):
        df = pd.DataFrame({"x": series, "y": y}).dropna()
        df["decile"] = pd.qcut(df["x"], 10, labels=False, duplicates="drop")
        return df.groupby("decile")["y"].mean().reset_index()

    plt.figure()
    for col in ["R_min", "R_avg"]:
        grp = decile_curve(merged[col], merged["FAIL"])
        plt.plot(grp["decile"], grp["y"], marker="o", label=col)
    plt.xlabel("Decile (higher score = better)")
    plt.ylabel("Failure rate")
    plt.title("Failure rate across deciles of R_min and R_avg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"failure_rate_deciles_Rmin_Ravg.png", dpi=180)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not create plot: {e}")

def two_prop_test(k1, n1, k2, n2):
    p1 = k1 / max(n1, 1)
    p2 = k2 / max(n2, 1)
    p = (k1 + k2) / max(n1 + n2, 1)
    se = sqrt(max(p*(1-p)*(1/n1 + 1/n2), 1e-12))
    z = (p1 - p2) / se if se > 0 else np.nan
    from math import erf, sqrt as _sqrt
    pval = 2*(1 - 0.5*(1 + erf(abs(z)/_sqrt(2)))) if not np.isnan(z) else np.nan
    return p1, p2, z, pval

threshold_rows = []
for q in [10, 15, 20]:
    s = merged["completeness"]; y = merged["FAIL"]
    low_cut = np.nanpercentile(s, q)
    high_cut = np.nanpercentile(s, 100 - q)
    y_low = y[s <= low_cut]
    y_high = y[s >= high_cut]
    k1, n1 = int(y_low.sum()), int(y_low.shape[0])
    k2, n2 = int(y_high.sum()), int(y_high.shape[0])
    p1, p2, z, pval = two_prop_test(k1, n1, k2, n2)
    threshold_rows.append({
        "percentile_rule": f"Bottom {q}% vs Top {q}%",
        "cut_low_value": float(low_cut),
        "cut_high_value": float(high_cut),
        "n_low": n1, "n_high": n2,
        "fail_rate_low": p1, "fail_rate_high": p2,
        "diff_pp": p1 - p2, "z": z, "p_value": pval
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUT_DIR/"threshold_suggestions_completeness.csv", index=False)

def within_task_concordance(df: pd.DataFrame, metric_col: str):
    """For each task, compare all PASS-FAIL model pairs; count how often metric(PASS) > metric(FAIL)."""
    cons = ties = total_pairs = 0
    per_task = []
    for tid, g in df.groupby("task_id"):
        gP = g[g["FAIL"] == 0][["model_name", metric_col]]
        gF = g[g["FAIL"] == 1][["model_name", metric_col]]
        if gP.empty or gF.empty:
            continue
        pairs = c = t = 0
        for _, rp in gP.iterrows():
            for _, rf in gF.iterrows():
                mp = rp[metric_col]; mf = rf[metric_col]
                if pd.isna(mp) or pd.isna(mf):
                    continue
                pairs += 1
                if mp > mf: c += 1
                elif mp == mf: t += 1
        if pairs > 0:
            cons += c; ties += t; total_pairs += pairs
            per_task.append({
                "task_id": tid,
                "pairs": pairs,
                "concordant": c,
                "ties": t,
                "concordance_rate": (c + 0.5*t) / pairs
            })
    overall_rate = (cons + 0.5*ties) / total_pairs if total_pairs > 0 else np.nan
 
    wins = cons
    trials = cons + (total_pairs - cons - ties)
    p_one_sided = binomtest(wins, trials, 0.5, alternative="greater").pvalue if trials > 0 else np.nan
    return overall_rate, p_one_sided, total_pairs, pd.DataFrame(per_task).sort_values("concordance_rate", ascending=False)

conc_rows = []
example_paths = {}
for metric in ["completeness", "logic", "efficiency"]:
    rate, pval, pairs, per_task_tbl = within_task_concordance(merged, metric)
    conc_rows.append({"metric": metric, "pairwise_concordance": rate, "p_value": pval, "num_pairs": pairs})
    # Save top few illustrative tasks
    path = OUT_DIR / f"within_task_examples_{metric}.csv"
    per_task_tbl.head(12).to_csv(path, index=False)
    example_paths[metric] = str(path)

concord_df = pd.DataFrame(conc_rows)
concord_df.to_csv(OUT_DIR/"within_task_concordance_summary.csv", index=False)

print("\n=== SUMMARY (Deliverable 1) ===")
print(f"N pairs: {N_total_pairs}  |  PASS={N_pass}  FAIL={N_fail}")
print("\nOverall correlations (metrics_only_overall_correlations.csv):")
print(overall_corrs.sort_values("spearman_p")[["metric","spearman_r","spearman_p","spearman_fdr","pearson_pb","pearson_p","kendall_tau","kendall_p","n"]].to_string(index=False))

if HAVE_SM and len(logit_rows):
    print("\nLogistic with model fixed effects (metrics_only_logit.csv):")
    df_logit = pd.read_csv(OUT_DIR/"metrics_only_logit.csv")
    print(df_logit.sort_values("p_value").to_string(index=False))
else:
    print("\n[INFO] Logistic FE results not produced (statsmodels not installed).")

print("\nAUC/threshold summary (metrics_only_auc_thresholds.csv) written.")
print("\n=== SUMMARY (Deliverable 2) ===")
print("Threshold suggestions (threshold_suggestions_completeness.csv):")
print(threshold_df.to_string(index=False))

print("\n=== SUMMARY (Deliverable 3) ===")
print("Within-task concordance (within_task_concordance_summary.csv):")
print(concord_df.to_string(index=False))
for k,v in example_paths.items():
    print(f"  Examples for {k}: {v}")

print(f"\nDecile plot (if generated): {OUT_DIR/'failure_rate_deciles_Rmin_Ravg.png'}")
print("\nDone.")
