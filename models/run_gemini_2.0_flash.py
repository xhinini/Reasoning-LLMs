#!/usr/bin/env python3
"""
Script to run Gemini-2.0-Flash-Thinking model on code generation tasks.
"""

import os
import time
import json
import logging
import glob
from typing import Dict, List, Any, Optional

# Import Google Generative AI SDK
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiReasoningRunner:
    """Class to run Gemini-2.0-Flash-Thinking model for reasoning tasks."""
    
    def __init__(self):
        """Initialize Gemini reasoning runner."""
        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Initialize with API key authentication
        genai.configure(api_key=api_key)
        
        # Set model
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
        logger.info(f"Initialized GeminiReasoningRunner with model: {self.model_name}")
    
    def generate_completion(self, prompt):
        """Generate completion using Gemini-2.0-Flash-Thinking.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict with content, reasoning_content, error, and duration
        """
        start_time = time.time()
        
        try:
            # Create a model instance
            model = genai.GenerativeModel(model_name=self.model_name)
            
            # Generate content with the model
            response = model.generate_content(prompt)
            
            # Extract content
            content = response.text
            
            # Parse reasoning and solution
            reasoning_content = ""
            solution_content = content
            
            if "### Reasoning Process:" in content and "### Solution:" in content:
                parts = content.split('### Solution:')
                reasoning_part = parts[0].split('### Reasoning Process:')[1].strip()
                solution_part = parts[1].strip()
                
                reasoning_content = "### Reasoning Process:\n" + reasoning_part
                solution_content = "### Solution:\n" + solution_part
            # Fallback to old format if needed
            elif "### Conclusion" in content:
                parts = content.split('### Conclusion')
                reasoning_content = parts[0].strip()
                solution_content = "### Conclusion" + parts[1].strip()
            
            # Calculate duration
            duration = time.time() - start_time
            
            return {
                "content": solution_content,
                "reasoning_content": reasoning_content,
                "error": None,
                "duration": duration
            }
        
        except Exception as e:
            # Log error
            logger.error(f"Error generating completion: {str(e)}")
            
            # Calculate duration
            duration = time.time() - start_time
            
            return {
                "content": "",
                "reasoning_content": "",
                "error": str(e),
                "duration": duration
            }


class PromptFormatter:
    """Class to format prompts for different task types."""
    
    def format_code_generation_prompt(self, sample):
        """Format prompt for code generation task."""
        task_id = sample.get("task_id", "Unknown")
        problem_description = sample.get("prompt", "No problem description provided.")
        
        prompt = f"""# Code Generation Task (ID: {task_id})

## Problem Description
{problem_description}

Please carefully study this software engineering problem, conduct a comprehensive analysis, and provide a solution with your reasoning process.

As an expert software developer, your task is to:
1. Understand the requirements 
2. Design an approach to solve the problem
3. Implement the solution in code
4. Verify the correctness of your solution

Format to Follow:

### Reasoning Process:
[Please explain your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]

<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Solution:
[Provide your complete code implementation here. Ensure it is functional, efficient, and addresses all requirements.]"""
        
        return {"prompt": prompt}
    
    def format_issue_resolution_prompt(self, sample):
        """Format prompt for issue resolution task."""
        task_id = sample.get("task_id", "Unknown")
        problem_description = sample.get("prompt", "No problem description provided.")
        
        prompt = f"""# Issue Resolution Task (ID: {task_id})

## Problem Description
{problem_description}

Please analyze this software issue, identify the root cause, and provide a solution.

## Instructions
As an expert software developer with years of experience in resolving complex issues. Please solve this problem step by step, showing your reasoning. First analyze the issue, then identify the root cause, and finally implement a solution. Please explain your reasoning process thoroughly.

When you're done with your analysis and solution, please include a section titled "### Conclusion" that contains only your final code solution."""
        
        return {"prompt": prompt}
    
    def format_code_translation_prompt(self, sample):
        """Format prompt for code translation task."""
        task_id = sample.get("task_id", "Unknown")
        problem_description = sample.get("prompt", "No problem description provided.")
        
        prompt = f"""# Code Translation Task (ID: {task_id})

## Problem Description
{problem_description}

Please translate the provided code to the target language while preserving its functionality.

## Instructions
As an expert software developer with years of experience in multiple programming languages. Please translate this code step by step, showing your reasoning. First analyze the source code, then plan your translation approach, and finally implement the translated code. Please explain your reasoning process thoroughly.

When you're done with your analysis and translation, please include a section titled "### Conclusion" that contains only your final translated code."""
        
        return {"prompt": prompt}


def run_evaluation(task_type, samples_dir, results_dir, sample_limit=None):
    """Run evaluation on samples.
    
    Args:
        task_type: Type of task to evaluate
        samples_dir: Directory containing dataset samples
        results_dir: Directory to save results
        sample_limit: Maximum number of samples to process (None for all)
    """
    # Initialize runner and formatter
    runner = GeminiReasoningRunner()
    formatter = PromptFormatter()
    
    # Get appropriate format function
    if task_type == "code_generation" or task_type == "code_generation_bcb_simple":
        format_func = formatter.format_code_generation_prompt
    elif task_type == "issue_resolution":
        format_func = formatter.format_issue_resolution_prompt
    elif task_type == "code_translation":
        format_func = formatter.format_code_translation_prompt
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Get samples
    if task_type == "code_generation":
        sample_pattern = os.path.join(samples_dir, "bigcodebench_*_samples.jsonl")
        sample_files = glob.glob(sample_pattern)
        if not sample_files:
            raise FileNotFoundError(f"No sample files found matching pattern: {sample_pattern}")
        sample_file = sample_files[0]  # Use the first matching file
    elif task_type == "code_generation_bcb_simple":
        sample_file = os.path.join(samples_dir, "bigcodebench_v0.1.0_hf_instruct_full_samples.jsonl")
    elif task_type == "issue_resolution":
        sample_file = os.path.join(samples_dir, "swe-bench-lite_samples.jsonl")
    elif task_type == "code_translation":
        sample_file = os.path.join(samples_dir, "codetransocean_samples.jsonl")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Check if sample file exists
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    
    # Load samples
    samples = []
    with open(sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Limit samples if specified
    if sample_limit is not None:
        samples = samples[:sample_limit]
    
    logger.info(f"Loaded {len(samples)} samples for {task_type} from {sample_file}")
    
    # Define output path
    output_path = os.path.join(results_dir, f"{task_type}_results.json")
    
    # Check if results file already exists
    results = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
                logger.info(f"Loaded {len(results)} existing results from {output_path}")
            except json.JSONDecodeError:
                logger.warning(f"Error loading existing results from {output_path}. Starting fresh.")
    
    # Get IDs of samples that have already been processed
    processed_ids = {result.get("sample_id") for result in results}
    
    # Filter out samples that have already been processed
    samples_to_process = [sample for sample in samples if sample.get("task_id") not in processed_ids]
    
    logger.info(f"Processing {len(samples_to_process)} new samples for {task_type}")
    
    # Process samples
    for i, sample in enumerate(samples_to_process):
        logger.info(f"Processing sample {i+1}/{len(samples_to_process)}: {sample.get('task_id', f'Sample_{i}')}")
        
        # Format prompt
        formatted_prompt = format_func(sample)
        prompt = formatted_prompt.get("prompt", "")
        
        # Generate completion
        completion = runner.generate_completion(prompt)
        
        # Create result object
        result = {
            "sample_id": sample.get("task_id", f"Sample_{i}"),
            "task_type": task_type,
            "prompt": prompt,
            "content": completion.get("content", ""),
            "reasoning_content": completion.get("reasoning_content", ""),
            "error": completion.get("error"),
            "duration": completion.get("duration"),
            "original_sample": sample
        }
        
        # Add to results
        results.append(result)
        
        # Save results after each sample (in case of interruption)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed sample {i+1}/{len(samples_to_process)}")
    
    logger.info(f"Evaluation completed for {task_type}. Results saved to {output_path}")

def main():
    """Main function to run Gemini-2.0-Flash-Thinking evaluation on dataset samples."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Gemini-2.0-Flash-Thinking evaluation on code samples")
    parser.add_argument("--samples_dir", type=str, default="dataset_samples",
                        help="Directory containing dataset samples")
    parser.add_argument("--results_dir", type=str, default="gemini_flash_results",
                        help="Directory to save results")
    parser.add_argument("--task", type=str, choices=["code_generation", "code_generation_bcb_simple", "issue_resolution", "code_translation", "all"],
                        default="all", help="Task type to evaluate")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Maximum number of samples to process per task (None for all)")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define task configurations with expected sample file patterns
    task_configs = {
        "code_generation": {"pattern": "bigcodebench_*_samples.jsonl"},
        "code_generation_bcb_simple": {"pattern": "bigcodebench_v0.1.0_hf_instruct_full_samples.jsonl"},
        # Currently not using these task types
        # "issue_resolution": {"pattern": "swe-bench*_samples.jsonl"},
        # "code_translation": {"pattern": "codetransocean_samples.jsonl"}
    }
    
    # Get actual sample files
    import glob
    
    # Run evaluation for specified task or all tasks
    if args.task == "all":
        for task in task_configs:
            run_evaluation(task, args.samples_dir, args.results_dir, args.sample_limit)
    else:
        run_evaluation(args.task, args.samples_dir, args.results_dir, args.sample_limit)

if __name__ == "__main__":
    main()
