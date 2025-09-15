import numpy as np
import pandas as pd
from pathlib import Path

def icc_oneway_random(X, nan_policy="omit"):
    """
    Parameters
    ----------
    X : array-like of shape (n_targets, k_raters)
        Ratings matrix. Each row = a task/target; each column = a rater.
        All rows must have the same number of raters k (after NaN handling).
    nan_policy : {"omit", "raise"}
        - "omit": drop any rows with NaN before computing.
        - "raise": raise a ValueError if NaNs are present.

    Returns
    -------
    out : dict
        {
          "ICC(1,1)": float,          # single-rater reliability
          "ICC(1,k)": float,          # reliability of the mean of k raters
          "MSR": float,               # between-target mean square
          "MSW": float,               # within-target (error) mean square
          "n": int,                   # number of targets used
          "k": int                    # raters per target
        }

    Notes
    -----
    Formulas (one-way ANOVA):
        grand   = mean(X)
        rowmean = mean over raters per target

        SSR = k * sum( (rowmean - grand)^2 )
        SST = sum( (X - grand)^2 )
        SSW = SST - SSR

        df_rows   = n - 1
        df_within = n * (k - 1)

        MSR = SSR / df_rows
        MSW = SSW / df_within

        ICC(1,1) = (MSR - MSW) / (MSR + (k - 1)*MSW)
        ICC(1,k) = (MSR - MSW) / MSR   = 1 - MSW/MSR
    """
    X = np.asarray(X, dtype=float)

    if np.isnan(X).any():
        if nan_policy == "omit":
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
        else:
            raise ValueError("NaNs present in X; set nan_policy='omit' to drop rows with NaNs.")

    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must be a 2D array with at least 2 targets and 2 raters.")

    n, k = X.shape

    # One-way ANOVA components
    grand = X.mean()
    row_means = X.mean(axis=1, keepdims=True)

    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_tot  = np.sum((X - grand) ** 2)
    ss_within = ss_tot - ss_rows

    df_rows = n - 1
    df_within = n * (k - 1)

    # Guard against degenerate cases
    if df_rows <= 0 or df_within <= 0:
        return {"ICC(1,1)": np.nan, "ICC(1,k)": np.nan, "MSR": np.nan, "MSW": np.nan, "n": n, "k": k}

    msr = ss_rows / df_rows
    msw = ss_within / df_within

    # ICCs (can be negative if MSR < MSW; that is allowed and indicates poor reliability)
    icc_11 = (msr - msw) / (msr + (k - 1) * msw) if (msr + (k - 1) * msw) != 0 else np.nan
    icc_1k = (msr - msw) / msr if msr != 0 else np.nan

    return {"ICC(1,1)": float(icc_11), "ICC(1,k)": float(icc_1k),
            "MSR": float(msr), "MSW": float(msw), "n": int(n), "k": int(k)}

xlsx = Path("icc_report.xlsx")
df = pd.read_excel(xlsx, sheet_name="per_task_dispersion")

def icc1k_over(df_group):
    X = df_group[["r1","r2","r3"]].dropna().to_numpy(float)
    return icc_oneway_random(X)

# Overall by dimension
rows = []
for dim, g in df.groupby("dimension"):
    stats = icc1k_over(g)
    rows.append({"dimension": dim, **stats})
icc_by_dim = pd.DataFrame(rows).sort_values("dimension")
print(icc_by_dim)

# By model × dimension
rows = []
for (model, dim), g in df.groupby(["model","dimension"]):
    stats = icc1k_over(g)
    rows.append({"model": model, "dimension": dim, **stats})
icc_by_model_dim = pd.DataFrame(rows).sort_values(["model","dimension"])
print(icc_by_model_dim)
