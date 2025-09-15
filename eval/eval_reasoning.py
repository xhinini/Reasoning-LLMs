import os
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI

IN_PATH = "o3_mini_cal.xlsx"
OUT_PATH = "o3_mini_cal_evaluations.xlsx"

class StepAssessment(BaseModel):
    index: int = Field(..., description="1-based index of the step")
    essential: bool = Field(..., description="Whether the step is essential for solving the problem")
    coherent_with_previous: bool = Field(..., description="Whether this step logically follows from previous steps")
    factually_correct: bool = Field(..., description="Whether this step is factually correct")
    justification: str = Field(..., description="Brief justification for the labels; keep concise")

class ProblemComponent(BaseModel):
    name: str
    addressed: bool
    notes: str = ""

class EfficiencyResult(BaseModel):
    total_steps: int
    essential_steps: int
    non_essential_steps: int
    score: float = Field(..., ge=0.0, le=1.0, description="Efficiency score = essential_steps / total_steps (0 if total_steps==0)")
    per_step: List[StepAssessment]

class LogicResult(BaseModel):
    coherent_steps: int
    incoherent_steps: int
    internal_contradictions: List[str] = []
    consistent_definitions: bool
    notes: str = ""
    score: float = Field(..., ge=0.0, le=1.0, description="Overall logic correctness and consistency score (0-1)")

class CompletenessResult(BaseModel):
    components: List[ProblemComponent]
    coverage_ratio: float = Field(..., ge=0.0, le=1.0, description="addressed components / total components (0 if none)")
    factual_issues: List[str] = []
    edge_cases_considered: bool
    notes: str = ""
    score: float = Field(..., ge=0.0, le=1.0, description="Overall completeness score (0-1)")

class ReasoningEvaluation(BaseModel):
    efficiency: EfficiencyResult
    logic_correctness_and_consistency: LogicResult
    completeness: CompletenessResult

SYSTEM_PROMPT = (
    "You are a careful reasoning evaluator. Follow the rubric strictly and return only a structured object. "
    "Do not include prose outside of the structured object."
)

USER_INSTRUCTIONS = (
    "Evaluate the reasoning steps according to three metrics: Efficiency; Logic Correctness and Consistency; "
    "Completeness.\n\n"
    "Guidelines:\n"
    "1) Efficiency: Identify individual reasoning steps (extract them if not explicitly numbered). Mark each as Essential if it introduces necessary information, "
    "performs required calculations, or makes crucial connections needed to solve the problem. Otherwise mark as Non-essential. "
    "Compute Efficiency score = essential_steps / total_steps (0 if total_steps==0).\n\n"
    "2) Logic Correctness and Consistency: For each step (except the first), mark whether it logically follows from previous steps. "
    "Identify any internal contradictions, definition inconsistencies, or approach shifts that invalidate earlier claims. "
    "Provide an overall Logic score on [0,1].\n\n"
    "3) Completeness: Identify all essential components and requirements in the problem. Mark each as addressed/not addressed in the reasoning. "
    "Note factual correctness issues and whether constraints/edge cases were considered. Provide an overall Completeness score on [0,1].\n\n"
    "Return a JSON that matches the provided schema exactly. Keep step justifications concise."
)

def find_reasoning_column(columns: List[str]) -> Optional[str]:
    lowered = [c.lower() for c in columns]
    
    for c in columns:
        if c.lower().strip() == "reasoning_content_openai-o3":
            return c
   
    for c in columns:
        cl = c.lower()
        if "reasoning" in cl and "content" in cl:
            return c
    return None


def evaluate_row(client: OpenAI, model: str, problem_description: str, reasoning_text: str) -> Optional[ReasoningEvaluation]:
    if not reasoning_text or not isinstance(reasoning_text, str):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTIONS + "\n\nProblem Description:\n" + problem_description + "\n\nReasoning Content:\n" + reasoning_text},
    ]

    try:
        resp = client.responses.parse(
            model=model,
            input=messages,
            text_format=ReasoningEvaluation,
        )
        parsed: ReasoningEvaluation = resp.output_parsed
        # Cross-validate simple ratios in case the model miscomputed
        # Recompute counts from per_step to ensure internal consistency
        if parsed.efficiency.per_step:
            essential_from_list = sum(1 for s in parsed.efficiency.per_step if s.essential)
            total_from_list = len(parsed.efficiency.per_step)
            parsed.efficiency.total_steps = total_from_list
            parsed.efficiency.essential_steps = essential_from_list
            parsed.efficiency.non_essential_steps = max(0, total_from_list - essential_from_list)
        # Recompute score from recomputed counts
        if parsed.efficiency.total_steps > 0:
            recomputed_eff = parsed.efficiency.essential_steps / parsed.efficiency.total_steps
        else:
            recomputed_eff = 0.0
        # Clamp float rounding
        parsed.efficiency.score = max(0.0, min(1.0, round(recomputed_eff, 4)))

        # Recompute logic coherent/incoherent counts from per-step flags when available
        try:
            total_steps = parsed.efficiency.total_steps
            if total_steps > 1 and parsed.efficiency.per_step:
                coherent = sum(1 for s in parsed.efficiency.per_step if s.index > 1 and s.coherent_with_previous)
                incoherent = max(0, (total_steps - 1) - coherent)
                parsed.logic_correctness_and_consistency.coherent_steps = coherent
                parsed.logic_correctness_and_consistency.incoherent_steps = incoherent
        except Exception:
            
            pass

        if parsed.completeness.components:
            addressed = sum(1 for comp in parsed.completeness.components if comp.addressed)
            coverage = addressed / len(parsed.completeness.components)
        else:
            coverage = 0.0
        parsed.completeness.coverage_ratio = max(0.0, min(1.0, round(coverage, 4)))

        return parsed
    except Exception as e:
        print(f"[warn] parse failed, task skipped: {e}")
        return None


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set; please export it before running.")
        return

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    xls = pd.ExcelFile(IN_PATH)
  
    only_sheets_env = os.getenv("SHEETS") 
    limit_rows_env = os.getenv("LIMIT_ROWS")
    only_sheets = None
    if only_sheets_env:
        only_sheets = {s.strip() for s in only_sheets_env.split(",") if s.strip()}
    limit_rows = None
    if limit_rows_env:
        try:
            limit_rows = int(limit_rows_env)
        except ValueError:
            print(f"[warn] LIMIT_ROWS is not an int: {limit_rows_env}")
            limit_rows = None

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        for sheet in xls.sheet_names:
            if only_sheets and sheet not in only_sheets:
                continue

            df = pd.read_excel(IN_PATH, sheet_name=sheet)
            if limit_rows is not None and limit_rows > 0:
                df = df.head(limit_rows)

            # Identify key columns
            problem_col = None
            for c in df.columns:
                if c.strip().lower() == "problem_description":
                    problem_col = c
                    break
            reasoning_col = find_reasoning_column(list(df.columns))

            if problem_col is None or reasoning_col is None:
                print(f"[info] Skipping sheet '{sheet}' due to missing columns. Found problem_col={problem_col}, reasoning_col={reasoning_col}")
                continue

            eval_rows = []
            for _, row in df.iterrows():
                task_id = row.get("task_id", None)
                problem = str(row.get(problem_col, "") or "")
                reasoning = str(row.get(reasoning_col, "") or "")

                evaluation = evaluate_row(client, model, problem, reasoning)
                if evaluation is None:
                    eval_rows.append({
                        "task_id": task_id,
                        "sheet": sheet,
                        "efficiency_score": None,
                        "logic_score": None,
                        "completeness_score": None,
                        "step_count": None,
                        "essential_count": None,
                        "coverage_ratio": None,
                        "contradictions_count": None,
                        "edge_cases_considered": None,
                        "evaluation_json": None,
                    })
                    continue

                eff = evaluation.efficiency
                logic = evaluation.logic_correctness_and_consistency
                comp = evaluation.completeness

                record = {
                    "task_id": task_id,
                    "sheet": sheet,
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
                print(record)
                eval_rows.append(record)

            out_df = pd.DataFrame(eval_rows)
            out_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Done. Wrote evaluations to {OUT_PATH}")


if __name__ == "__main__":
    main()
