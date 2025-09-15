import sys
from pathlib import Path
import pandas as pd
import numpy as np


EXCEL_PATH = "icc(1,k) (1).xlsx"  
OUT_DIR = Path("./rq1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_METRICS = ["efficiency", "logic", "completeness"]  

def find_sheet_case_insensitive(xls: pd.ExcelFile, target: str):
    """Return the exact sheet name matching target (case-insensitive), or None."""
    for name in xls.sheet_names:
        if name.strip().lower() == target.strip().lower():
            return name
    return None

def find_like_sheet(xls: pd.ExcelFile, substring: str):
    """Return the first sheet whose name contains substring (case-insensitive), or None."""
    for name in xls.sheet_names:
        if substring.lower() in name.lower():
            return name
    return None

def summarize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return pd.Series({"mean": np.nan, "sd": np.nan, "median": np.nan,
                          "q1": np.nan, "q3": np.nan, "n": 0})
    return pd.Series({
        "mean": float(np.mean(s)),
        "sd": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "median": float(np.median(s)),
        "q1": float(np.quantile(s, 0.25)),
        "q3": float(np.quantile(s, 0.75)),
        "n": int(len(s)),
    })

def format_row(mean, sd, median, q1, q3, n) -> str:
    if pd.isna(mean):
        return "—"
    return f"{mean:.3f}±{sd:.3f} [ {median:.3f}; {q1:.3f}–{q3:.3f} ] (n={int(n)})"

xls = pd.ExcelFile(EXCEL_PATH)

sheet_dim = find_sheet_case_insensitive(xls, "per_task_dimension")
if sheet_dim is None:
    raise RuntimeError("Sheet 'per_task_dimension' not found.")

df_dim = xls.parse(sheet_dim)

req_cols = {"task_id", "model", "dimension", "mean_of_3"}
missing = req_cols - set(df_dim.columns)
if missing:
    raise RuntimeError(f"Missing columns in '{sheet_dim}': {missing}")

df_wide = (
    df_dim[["task_id", "model", "dimension", "mean_of_3"]]
      .assign(dimension=lambda d: d["dimension"].astype(str).str.strip().str.lower())
      .pivot_table(index=["task_id", "model"], columns="dimension", values="mean_of_3", aggfunc="mean")
      .reset_index()
)


present_metrics = [m for m in EXPECTED_METRICS if m in df_wide.columns]
if len(present_metrics) == 0:
    raise RuntimeError(f"None of the expected metrics found after pivot: {EXPECTED_METRICS}")


sheet_hard = find_sheet_case_insensitive(xls, "hard") or find_like_sheet(xls, "hard")
if sheet_hard is None:
    
    df_wide["split"] = "All"
else:
    df_hard = xls.parse(sheet_hard)
    if not {"task_id", "model"}.issubset(df_hard.columns):
        raise RuntimeError(f"'hard' sheet must contain columns task_id and model (found: {df_hard.columns.tolist()})")
    hard_pairs = set(zip(df_hard["task_id"].astype(str), df_hard["model"].astype(str)))
    df_wide["split"] = ["Hard" if (str(t), str(m)) in hard_pairs else "Full"
                        for t, m in zip(df_wide["task_id"], df_wide["model"])]


# Compute mean of the available metrics per row
df_wide["RQI_row"] = df_wide[present_metrics].mean(axis=1)

# Aggregate per model × split
group_cols = ["model", "split"]
agg_frames = []
for metric in present_metrics + ["RQI_row"]:
    g = (
        df_wide.groupby(group_cols)[metric]
               .apply(summarize_series)
               .unstack(-1)  
               .reset_index()
    )
    g.columns = ["model", "split",
                 f"{metric}_mean", f"{metric}_sd", f"{metric}_median",
                 f"{metric}_q1", f"{metric}_q3", f"{metric}_n"]
    agg_frames.append(g)

from functools import reduce
rawstats = reduce(lambda L, R: pd.merge(L, R, on=["model", "split"], how="outer"), agg_frames)


rawstats_path = OUT_DIR / "rq1_overall_scores_rawstats.csv"
rawstats.to_csv(rawstats_path, index=False)

pairs = rawstats[["model", "split"]].drop_duplicates().sort_values(["model", "split"])

formatted_cols = {}
for metric in present_metrics + ["RQI_row"]:
    cols = [f"{metric}_mean", f"{metric}_sd", f"{metric}_median", f"{metric}_q1", f"{metric}_q3", f"{metric}_n"]
    sub = rawstats[["model", "split"] + cols].copy()
    sub["__fmt"] = sub.apply(lambda r: format_row(r[f"{metric}_mean"], r[f"{metric}_sd"],
                                                  r[f"{metric}_median"], r[f"{metric}_q1"],
                                                  r[f"{metric}_q3"], r[f"{metric}_n"]), axis=1)
    formatted_cols[metric] = sub[["model", "split", "__fmt"]].rename(columns={"__fmt": metric})

formatted = pairs.copy()
for metric, sub in formatted_cols.items():
    formatted = formatted.merge(sub, on=["model", "split"], how="left")

ordered_cols = ["model", "split"] + [m for m in EXPECTED_METRICS if m in formatted.columns] + ["RQI_row"]
formatted = formatted[ordered_cols]

formatted_path = OUT_DIR / "rq1_overall_scores_formatted.csv"
formatted.to_csv(formatted_path, index=False)

print("Wrote:", rawstats_path)
print("Wrote:", formatted_path)
print("\nPresent metrics:", present_metrics)
print("Rows in per_task×model:", len(df_wide))
print("Models:", df_wide['model'].nunique(), "Splits:", df_wide['split'].unique().tolist())
