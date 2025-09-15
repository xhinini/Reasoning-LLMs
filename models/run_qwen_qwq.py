#!/usr/bin/env python3
"""
Script to run Qwen-QwQ reasoning model on code generation tasks using DashScope API.
"""

import os
import time
import json
import logging
import glob
import re
from typing import Dict, List, Any, Optional

# Import OpenAI client for OpenRouter API
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qwen_qwq_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QwQReasoningRunner:
    """Class to run Qwen-QwQ model for reasoning tasks using OpenRouter API."""
    
    def __init__(self, reasoning_effort="medium"):
        """Initialize Qwen-QwQ reasoning runner.
        
        Args:
            reasoning_effort: Effort level for reasoning ("low", "medium", or "high")
        """
        # Get API key from environment variable
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Set model name
        self.model_name = "qwen/qwq-32b:free"
        
        # Set site info for OpenRouter
        self.site_url = os.environ.get("SITE_URL", "https://github.com/user/llm-reasoning")
        self.site_name = os.environ.get("SITE_NAME", "LLM-Reasoning-Project")
        
        # Set a fixed max_tokens value of 4096
        self.max_tokens = 4096
        
        # Note: If max_tokens is not specified in the API call,
        # OpenRouter will use the model's default setting

        # Set generation parameters
        self.temperature = 0.6    
        self.top_p = 0.95        
        self.top_k = 30           
        self.min_p = 0.0          
        self.presence_penalty = 1.0  
        
        logger.info(f"Initialized QwQReasoningRunner with reasoning_effort: {reasoning_effort}")
    
    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        """Generate completion using Qwen-QwQ via OpenRouter API.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict with content, reasoning_content, error, and duration
        """
        start_time = time.time()
        
        try:
            # Format messages for the chat API
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,  # Set to 4096
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name
                },
                extra_body={},
                stream=False  # Set to False for non-streaming response
            )
            
            # Extract content and reasoning_content
            content = response.choices[0].message.content
            reasoning_content = ""
            
            # Check if reasoning_content is available in the response
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content
            # If not, try to extract it from the content using our custom format
            elif "### Reasoning Process:" in content and "### Solution:" in content:
                parts = content.split('### Solution:')
                reasoning_part = parts[0].split('### Reasoning Process:')[1].strip()
                solution_part = parts[1].strip()
                
                reasoning_content = "### Reasoning Process:\n" + reasoning_part
                content = "### Solution:\n" + solution_part
            # Also check for QwQ's native format with <think> tags
            elif "<think>" in content and "</think>" in content:
                parts = content.split("</think>", 1)
                reasoning_part = parts[0].split("<think>", 1)[1].strip()
                solution_part = parts[1].strip()
                
                reasoning_content = "<think>\n" + reasoning_part + "\n</think>"
                content = solution_part
            
            # Calculate duration
            duration = time.time() - start_time
            
            return {
                "content": content,
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
    
    def _extract_reasoning_content(self, response, content):
        """Extract reasoning content from response.
        
        Args:
            response: Response object
            content: Content string
            
        Returns:
            Reasoning content string
        """
        # QwQ model uses <think> and </think> tags for reasoning
        if "<think>" in content and "</think>" in content:
            parts = content.split("</think>", 1)
            reasoning_part = parts[0].split("<think>", 1)[1].strip()
            return "<think>\n" + reasoning_part + "\n</think>"
        
        # Also check for our custom format with "### Reasoning Process:" and "### Solution:"
        elif "### Reasoning Process:" in content and "### Solution:" in content:
            parts = content.split('### Solution:')
            reasoning_part = parts[0].split('### Reasoning Process:')[1].strip()
            return "### Reasoning Process:\n" + reasoning_part
        
        # If no structured reasoning format is found, return empty string
        return ""
    
    def _split_reasoning_steps(self, reasoning_content):
        """Split reasoning content into steps.
        
        Args:
            reasoning_content: Reasoning content string
            
        Returns:
            List of reasoning steps
        """
        # If no reasoning content, return empty list
        if not reasoning_content:
            return []
        
        # Try to split by step markers like "<step 1>", "<step 2>", etc.
        import re
        steps = re.findall(r'<step \d+>.*?(?=<step \d+>|$)', reasoning_content, re.DOTALL)
        
        # If no steps found, try splitting by newlines
        if not steps:
            steps = reasoning_content.split('\n\n')
        
        return steps


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


def run_evaluation(task_type, samples_dir, results_dir, reasoning_effort="medium", sample_limit=None, skip=None):
    """Run evaluation on samples.
    
    Args:
        task_type: Type of task to evaluate
        samples_dir: Directory containing dataset samples
        results_dir: Directory to save results
        reasoning_effort: Effort level for reasoning ("low", "medium", or "high")
        sample_limit: Maximum number of samples to process (None for all)
        skip: Number of samples to skip
    """
    # Initialize runner and formatter
    runner = QwQReasoningRunner(reasoning_effort=reasoning_effort)
    formatter = PromptFormatter()
    
    # Get appropriate format function
    if task_type == "code_generation" or task_type == "full":
        format_func = formatter.format_code_generation_prompt
        sample_pattern = os.path.join(samples_dir, "humanevalpro_mbpppro_hard_samples.jsonl")
    elif task_type == "code_generation_bcb_simple":
        format_func = formatter.format_code_generation_prompt
        sample_pattern = os.path.join(samples_dir, "bigcodebench_v0.1.0_hf_instruct_samples.jsonl")
    elif task_type == "issue_resolution":
        format_func = formatter.format_issue_resolution_prompt
        sample_pattern = os.path.join(samples_dir, "swe-bench*_samples.jsonl")
    elif task_type == "code_translation":
        format_func = formatter.format_code_translation_prompt
        sample_pattern = os.path.join(samples_dir, "codetransocean_samples.jsonl")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Get sample file
    sample_files = glob.glob(sample_pattern)
    if not sample_files:
        raise FileNotFoundError(f"No sample files found for pattern: {sample_pattern}")
    sample_file = sample_files[0]
    
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
    output_path = os.path.join(results_dir, f"qwq_{task_type}_results.json")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
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
    if skip is None:
        samples_to_process = [sample for sample in samples if sample.get("task_id") not in processed_ids]
    else:
        # Skip the first 'skip' samples
        samples_to_process = samples[skip:]
    
    logger.info(f"Processing {len(samples_to_process)} new samples for {task_type}")
    
    # Process samples
    for i, sample in enumerate(samples_to_process):
        sample_id = sample.get("task_id", f"Sample_{i}")
        
        # Skip if already processed
        if sample_id in processed_ids:
            logger.info(f"Skipping already processed sample: {sample_id}")
            continue
        
        logger.info(f"Processing sample {i+1}/{len(samples_to_process)}: {sample_id}")
        
        # Format prompt
        formatted_prompt = format_func(sample)
        prompt = formatted_prompt.get("prompt", "")
        
        # Generate completion
        completion = runner.generate_completion(prompt)
        
        # Create result object
        result = {
            "sample_id": sample_id,
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
    """Main function to run Qwen-QwQ evaluation on dataset samples."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Qwen-QwQ evaluation on code samples")
    parser.add_argument("--samples_dir", type=str, default="dataset_samples",
                        help="Directory containing dataset samples")
    parser.add_argument("--results_dir", type=str, default="qwq_results",
                        help="Directory to save results")
    parser.add_argument("--task", type=str, choices=["code_generation", "code_generation_bcb_simple", "issue_resolution", "code_translation", "all"],
                        default="code_generation_bcb_simple", help="Task type to evaluate")
    parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"],
                        default="medium", help="Effort level for reasoning")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Maximum number of samples to process (None for all)")
    parser.add_argument("--skip", type=int, default=None,
                        help="Number of samples to skip")
    
    args = parser.parse_args()
    
    # Run evaluation for specified task(s)
    if args.task == "all":
        tasks_to_run = ["code_generation", "code_generation_bcb_simple", "issue_resolution", "code_translation"]
    else:
        tasks_to_run = [args.task]
    
    for task_type in tasks_to_run:
        logger.info(f"Starting evaluation for {task_type}")
        run_evaluation(
            task_type,
            args.samples_dir,
            args.results_dir,
            args.reasoning_effort,
            args.sample_limit,
            args.skip
        )
    
    logger.info("All evaluations completed")


if __name__ == "__main__":
    main()
