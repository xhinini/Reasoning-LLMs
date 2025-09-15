import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from openai import OpenAI

import evaluate_reasoning as er

DEFAULT_IN = "problematic_reasonings.xlsx"
REFINED_REASONINGS = "refined_reasonings.xlsx"
REFINED_EVALS = "refined_evaluations.xlsx"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in df.columns}
    for name in candidates:
        for c in df.columns:
            if c.lower().strip() == name:
                return c
    return None


def pick_problem_col(df: pd.DataFrame) -> Optional[str]:
  
    return find_col(df, ["problem_description", "problem", "description"]) or None


def pick_reasoning_col(df: pd.DataFrame) -> Optional[str]:

    explicit = find_col(df, ["reasoning_content_openai-o3", "reasoning_content", "reasoning"])
    if explicit:
        return explicit
   
    for c in df.columns:
        cl = c.lower()
        if "reasoning" in cl and "content" in cl:
            return c
    return None


def pick_metric_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["metric", "metric_phrase", "note", "comment"]) or None


GUIDED_SYSTEM = (
    "You are revising a chain-of-thought to be more effective at solving the problem. "
    "Write a self-contained, numbered reasoning (e.g., <step 1>, <step 2>, ...). "
    "Keep steps focused, logically connected, and minimal yet sufficient to reach the solution."
)

UNGUIDED_SYSTEM = GUIDED_SYSTEM

GUIDED_USER_TMPL = (
    "Problem description:\n{problem}\n\n"
    "Original reasoning (needs improvement):\n{original}\n\n"
    "High-level guidance: {metric}.\n\n"
    "Please produce an improved reasoning only (no code), numbered steps, concise and logically coherent."
)

UNGUIDED_USER_TMPL = (
    "Problem description:\n{problem}\n\n"
    "Original reasoning (needs improvement):\n{original}\n\n"
    "Please self-review and produce an improved reasoning only (no code), numbered steps, concise and logically coherent."
)


def correct_reasoning(client: OpenAI, model: str, problem: str, original: str, metric_phrase: Optional[str], guided: bool) -> str:
    if guided:
        metric = metric_phrase or "This reasoning has problems; you need to fix it to better solve the problem."
        user = GUIDED_USER_TMPL.format(problem=problem, original=original, metric=metric)
        system = GUIDED_SYSTEM
    else:
        user = UNGUIDED_USER_TMPL.format(problem=problem, original=original)
        system = UNGUIDED_SYSTEM

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # Extract text content
    try:
        # responses API: aggregate text from output
        if hasattr(resp, "output_text"):
            return resp.output_text
        # Fallback: parse content pieces
        chunks = []
        for o in getattr(resp, "output", []) or []:
            if o.get("type") == "message":
                for c in o.get("content", []) or []:
                    if c.get("type") == "text":
                        chunks.append(c.get("text", ""))
        return "\n".join([c for c in chunks if c])
    except Exception:
        # last resort
        return str(resp)


def evaluate_rows(df: pd.DataFrame, sheet_name: str, client_eval: OpenAI, eval_model: str, problem_col: str, reasoning_col: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        task_id = row.get("task_id", None)
        problem = str(row.get(problem_col, "") or "")
        reasoning = str(row.get(reasoning_col, "") or "")
        evaluation = er.evaluate_row(client_eval, eval_model, problem, reasoning)
        if evaluation is None:
            rec = {
                "task_id": task_id,
                "sheet": sheet_name,
                "efficiency_score": None,
                "logic_score": None,
                "completeness_score": None,
                "step_count": None,
                "essential_count": None,
                "coverage_ratio": None,
                "contradictions_count": None,
                "edge_cases_considered": None,
                "evaluation_json": None,
            }
        else:
            eff = evaluation.efficiency
            logic = evaluation.logic_correctness_and_consistency
            comp = evaluation.completeness
            rec = {
                "task_id": task_id,
                "sheet": sheet_name,
                "efficiency_score": eff.score,
                "logic_score": logic.score,
                "completeness_score": comp.score,
                "step_count": eff.total_steps,
                "essential_count": eff.essential_steps,
                "coverage_ratio": comp.coverage_ratio,
                "contradictions_count": len(logic.internal_contradictions) if logic.internal_contradictions else 0,
                "edge_cases_considered": comp.edge_cases_considered,
                "evaluation_json": evaluation.model_dump_json(),
            }
        records.append(rec)
    return pd.DataFrame(records)


def _load_medium_reasonings(o3_path: Path) -> pd.DataFrame:
    """Load medium-level reasonings from o3_mini_cal.xlsx (sheets: medium_hard, medim_full)."""
    xls = pd.ExcelFile(o3_path)
    medium_sheets = [s for s in xls.sheet_names if s.lower().startswith("medium") or s.lower().startswith("medim")]
    frames = []
    for sh in medium_sheets:
        df = pd.read_excel(o3_path, sheet_name=sh)
        if df.empty:
            continue
        # locate columns
        prob_col = pick_problem_col(df) or "problem_description"
        reas_col = pick_reasoning_col(df)
        if not reas_col:
            continue
        sub = df[[c for c in ["task_id", prob_col, reas_col] if c in df.columns]].copy()
        sub = sub.rename(columns={prob_col: "problem_description", reas_col: "reasoning_content_openai-o3"})
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["task_id","problem_description","reasoning_content_openai-o3"])
    out = pd.concat(frames, ignore_index=True)
    # Drop duplicates keeping first
    out = out.dropna(subset=["task_id"]).drop_duplicates(subset=["task_id"]) 
    return out


def _metric_phrase_from_flags(has_eff: bool, has_log: bool, has_comp: bool) -> str:
    issues = []
    if has_eff:
        issues.append("Efficiency")
    if has_log:
        issues.append("Logic Correctness and Consistency")
    if has_comp:
        issues.append("Completeness")
    if not issues:
        return "This reasoning has problems and you need to fix it to better solve the problem."
    if len(issues) == 1:
        joined = issues[0]
    elif len(issues) == 2:
        joined = f"{issues[0]} and {issues[1]}"
    else:
        joined = f"{issues[0]}; {issues[1]} and {issues[2]}"
    return (
        "This reasoning has problems and you need to fix it to be able to better solve the problem. "
        f"The reasoning has issues with {joined}."
    )


def build_problematic_from_scores(score_path: Path, o3_path: Path, threshold: float = 1.0, topk: int = 0) -> pd.DataFrame:
    """Build a problematic reasoning DataFrame from the human score sheet for model2 (medium).

    - Reads Sheet1 from score_sheet.xlsx
    - Computes mean across raters for model2_{efficiency,logic,completeness}
    - Selects tasks where any mean < threshold
    - Joins to medium reasonings from o3_mini_cal.xlsx using task_id
    - Produces columns: task_id, problem_description, reasoning_content_openai-o3, metric
    """
    if not score_path.exists():
        raise FileNotFoundError(f"Score sheet not found: {score_path}")
    if not o3_path.exists():
        raise FileNotFoundError(f"o3 workbook not found: {o3_path}")
    scores = pd.read_excel(score_path, sheet_name="Sheet1")
    if "task_id" not in scores.columns:
        raise ValueError("score_sheet.xlsx must contain 'task_id' column")

    # Identify rater columns for model2
    # Case-insensitive matching using a mapping
    col_map = {c: c.lower() for c in scores.columns}
    def cols_for(metric: str) -> List[str]:
        pat = re.compile(rf"^r\d+_model2_{metric}$")
        return [c for c in scores.columns if pat.match(col_map[c])]

    eff_cols = cols_for("efficiency")
    log_cols = cols_for("logic")
    comp_cols = cols_for("completeness")
    # Compute means ignoring NaNs
    def mean_across(cols):
        return scores[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1) if cols else pd.Series(np.nan, index=scores.index)

    scores["model2_eff_mean"] = mean_across(eff_cols)
    scores["model2_log_mean"] = mean_across(log_cols)
    scores["model2_comp_mean"] = mean_across(comp_cols)
    scores["model2_min_mean"] = scores[["model2_eff_mean","model2_log_mean","model2_comp_mean"]].min(axis=1)

    # Flag non-perfect
    scores["flag_eff"] = scores["model2_eff_mean"].lt(threshold)
    scores["flag_log"] = scores["model2_log_mean"].lt(threshold)
    scores["flag_comp"] = scores["model2_comp_mean"].lt(threshold)
   
    problematic = scores[(scores["flag_eff"]) | (scores["flag_log"]) | (scores["flag_comp"])].copy()
    candidates = problematic if not problematic.empty else scores.copy()

    medium_df = _load_medium_reasonings(o3_path)
    merged = candidates.merge(medium_df, on="task_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["task_id","problem_description","reasoning_content_openai-o3","metric"])

   
    if isinstance(topk, int) and topk > 0:
        merged = merged.sort_values("model2_min_mean", ascending=True).head(topk)

    merged["metric"] = merged.apply(lambda r: _metric_phrase_from_flags(bool(r["flag_eff"]), bool(r["flag_log"]), bool(r["flag_comp"])), axis=1)
    out = merged[["task_id","problem_description","reasoning_content_openai-o3","metric"]].copy()
    return out


def run_pipeline(in_path: Path, out_reasonings: Path, out_evals: Path, run_analysis: bool):
    if not in_path.exists():
        raise FileNotFoundError(f"Input Excel not found: {in_path}")


    correct_model = os.getenv("OPENAI_CORRECT_MODEL", os.getenv("OPENAI_MODEL", "o3-mini"))
    eval_model = os.getenv("OPENAI_MODEL", "gpt-5-nano")


    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client_correct = OpenAI()
    client_eval = OpenAI()

    xls = pd.ExcelFile(in_path)
    guided_frames = []
    unguided_frames = []

    for sh in xls.sheet_names:
        src = pd.read_excel(in_path, sheet_name=sh)
        if src.empty:
            continue
        problem_col = pick_problem_col(src)
        reasoning_col = pick_reasoning_col(src)
        metric_col = pick_metric_col(src)
        if not problem_col or not reasoning_col:
            print(f"[warn] Skipping sheet '{sh}' due to missing columns. problem_col={problem_col}, reasoning_col={reasoning_col}")
            continue

        guided_rows = []
        unguided_rows = []
        print(f"Processing sheet '{sh}' with {len(src)} rows")
        for _, row in src.iterrows():
            task_id = row.get("task_id", None)
            problem = str(row.get(problem_col, "") or "")
            original = str(row.get(reasoning_col, "") or "")
            metric_phrase = str(row.get(metric_col, "") or "") if metric_col else ""
            print(f"Processing row {task_id}")
            # Guided correction
            guided_text = correct_reasoning(client_correct, correct_model, problem, original, metric_phrase, guided=True)
            guided_rows.append({
                "task_id": task_id,
                "problem_description": problem,
                "reasoning_content_openai-o3": guided_text,
                "source_sheet": sh,
                "ablation": "guided",
                "metric_phrase": metric_phrase,
            })

            # Unguided correction
            unguided_text = correct_reasoning(client_correct, correct_model, problem, original, metric_phrase=None, guided=False)
            unguided_rows.append({
                "task_id": task_id,
                "problem_description": problem,
                "reasoning_content_openai-o3": unguided_text,
                "source_sheet": sh,
                "ablation": "unguided",
            })
            print(f"Processed row {task_id}")

        guided_df = pd.DataFrame(guided_rows)
        unguided_df = pd.DataFrame(unguided_rows)
        guided_frames.append(guided_df)
        unguided_frames.append(unguided_df)


    guided_all = pd.concat(guided_frames, ignore_index=True) if guided_frames else pd.DataFrame()
    unguided_all = pd.concat(unguided_frames, ignore_index=True) if unguided_frames else pd.DataFrame()
    print(f"Guided: {len(guided_all)} rows")
    print(f"Unguided: {len(unguided_all)} rows")
    print(f"Wrote refined reasonings to {out_reasonings}")
    with pd.ExcelWriter(out_reasonings, engine="openpyxl") as writer:
        if not guided_all.empty:
            guided_all.to_excel(writer, sheet_name="guided", index=False)
        if not unguided_all.empty:
            unguided_all.to_excel(writer, sheet_name="unguided", index=False)

    eval_frames = []
    if not guided_all.empty:
        eval_guided = evaluate_rows(guided_all, "guided", client_eval, eval_model, "problem_description", "reasoning_content_openai-o3")
        eval_frames.append(("guided", eval_guided))
    if not unguided_all.empty:
        eval_unguided = evaluate_rows(unguided_all, "unguided", client_eval, eval_model, "problem_description", "reasoning_content_openai-o3")
        eval_frames.append(("unguided", eval_unguided))

    with pd.ExcelWriter(out_evals, engine="openpyxl") as writer:
        for name, df in eval_frames:
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"Wrote refined evaluations to {out_evals}")

    if run_analysis:
        import subprocess
        env = os.environ.copy()
        env["EVAL_PATH"] = str(Path(out_evals).resolve())
        env["OUT_SUMMARY"] = str(Path("refined_summary.xlsx").resolve())
        try:
            subprocess.run(["python", "analyze_evaluations.py"], check=False, env=env)
        except Exception as e:
            print(f"[warn] Failed to run analysis: {e}")


def main():
    parser = argparse.ArgumentParser(description="Refine problematic reasonings with o3-mini, evaluate, and analyze (with ablation).")
    parser.add_argument("--input", type=str, default=DEFAULT_IN, help="Path to Excel with problematic reasonings")
    parser.add_argument("--score-sheet", type=str, default=None, help="Path to score_sheet.xlsx containing human scores")
    parser.add_argument("--o3", type=str, default="o3_mini_cal.xlsx", help="Path to o3_mini_cal.xlsx")
    parser.add_argument("--threshold", type=float, default=1.0, help="Non-perfect threshold for selecting problematic items (default: < 1.0)")
    parser.add_argument("--topk", "-topk", type=int, default=0, help="Select only the K worst model2 tasks by min(mean_eff, mean_log, mean_comp)")
    parser.add_argument("--run-analysis", action="store_true", help="Run analyze_evaluations.py after evaluation (temporary file swap)")
    args = parser.parse_args()

    input_path = Path(args.input)
    
    if args.score_sheet:
        built = build_problematic_from_scores(Path(args.score_sheet), Path(args.o3), threshold=args.threshold, topk=args.topk)
        if built.empty:
            print("[info] No problematic items selected from score sheet under given threshold; nothing to refine.")
            return
       
        with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
            built.to_excel(writer, sheet_name="medium", index=False)
        print(f"Built problematic set from scores: {len(built)} rows -> {input_path}")

    run_pipeline(input_path, Path(REFINED_REASONINGS), Path(REFINED_EVALS), run_analysis=args.run_analysis)


if __name__ == "__main__":
    main()
