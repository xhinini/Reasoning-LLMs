#!/usr/bin/env python3
"""
Script to run Gemini-2.5-Flash model with thinking capabilities on code generation tasks.
"""

import os
import time
import json
import logging
import glob
import re
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import OpenAI client for OpenRouter API
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_flash_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeminiFlashRunner:
    """Class to run Gemini-2.5-Flash model for reasoning tasks using OpenRouter API."""
    
    def __init__(self):
        """Initialize Gemini Flash runner."""
        # Get API key from environment variable
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Set model name with thinking capabilities
        self.model_name = "google/gemini-2.5-flash-preview:thinking"
        
        # Set site info for OpenRouter
        self.site_url = os.environ.get("SITE_URL", "https://github.com/user/llm-reasoning")
        self.site_name = os.environ.get("SITE_NAME", "LLM-Reasoning-Project")
        
        # Set a fixed max_tokens value
        self.max_tokens = 4096
        
        # Set generation parameters
        self.temperature = 0.6
        self.top_p = 0.95
        
        logger.info(f"Initialized GeminiFlashRunner")
    
    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        """Generate completion using Gemini-2.5-Flash via OpenRouter API.
        
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name
                },
                extra_body={},
                stream=False
            )
            
            # Extract content
            content = response.choices[0].message.content
            
            # Extract reasoning content
            reasoning_content = self._extract_reasoning_content(content)
            
            # Calculate duration
            duration = time.time() - start_time
            
            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "error": None,
                "duration": duration
            }
            
        except Exception as e:
            # Handle errors
            logger.error(f"Error generating completion: {str(e)}")
            duration = time.time() - start_time
            
            # Add a delay if we hit rate limits
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning("Rate limit or quota exceeded. Waiting 60 seconds before continuing...")
                time.sleep(60)
            
            return {
                "content": "",
                "reasoning_content": "",
                "error": f"Error: {str(e)}",
                "duration": duration
            }
    
    def _extract_reasoning_content(self, content: str) -> str:
        """Extract reasoning content from response.
        
        Args:
            content: Content string
            
        Returns:
            Reasoning content string
        """
        # Check for thinking format (specific to models with :thinking suffix)
        if "<thinking>" in content and "</thinking>" in content:
            parts = content.split("</thinking>", 1)
            thinking_part = parts[0].split("<thinking>", 1)[1].strip()
            return thinking_part
        
        # Check for our custom format
        elif "### Reasoning Process:" in content and "### Solution:" in content:
            parts = content.split('### Solution:')
            reasoning_part = parts[0].split('### Reasoning Process:')[1].strip()
            return "### Reasoning Process:\n" + reasoning_part
        
        # No reasoning content found
        return ""


class PromptFormatter:
    """Class to format prompts for different task types."""
    
    def format_code_generation_prompt(self, sample):
        """Format prompt for code generation task."""
        task_id = sample.get("task_id", "Unknown")
        problem_description = sample.get("prompt", "")
        
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
        problem_description = sample.get("prompt", "")
        
        prompt = f"""# Issue Resolution Task (ID: {task_id})

## Problem Description
{problem_description}

Please analyze this software issue, identify the root cause, and provide a solution with your reasoning process.

Format to Follow:

### Reasoning Process:
[Please explain your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]

<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Solution:
[Provide your complete solution here. Ensure it addresses the root cause of the issue.]"""
        
        return {"prompt": prompt}
    
    def format_code_translation_prompt(self, sample):
        """Format prompt for code translation task."""
        task_id = sample.get("task_id", "Unknown")
        problem_description = sample.get("prompt", "")
        
        prompt = f"""# Code Translation Task (ID: {task_id})

## Problem Description
{problem_description}

Please translate the provided code to the target language, maintaining its functionality and structure.

Format to Follow:

### Reasoning Process:
[Please explain your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]

<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Solution:
[Provide your complete translated code here. Ensure it maintains the functionality of the original code.]"""
        
        return {"prompt": prompt}


def run_evaluation(task_type, samples_dir, results_dir, sample_limit=None, skip=None):
    """Run evaluation on samples.
    
    Args:
        task_type: Type of task to evaluate
        samples_dir: Directory containing dataset samples
        results_dir: Directory to save results
        sample_limit: Maximum number of samples to process (None for all)
        skip: Number of samples to skip
    """
    # Initialize runner and formatter
    runner = GeminiFlashRunner()
    formatter = PromptFormatter()
    
    # Determine format function based on task type
    if task_type == "code_generation":
        format_func = formatter.format_code_generation_prompt
        sample_pattern = os.path.join(samples_dir, "humanevalpro_mbpppro_hard_samples.jsonl")
    elif task_type == "code_generation_bcb_hard":
        format_func = formatter.format_code_generation_prompt
        sample_pattern = os.path.join(samples_dir, "bigcodebench_v0.1.0_hf_instruct_samples.jsonl")
    elif task_type == "code_generation_bcb_full":
        format_func = formatter.format_code_generation_prompt
        sample_pattern = os.path.join(samples_dir, "bigcodebench_v0.1.0_hf_instruct_full_samples.jsonl")
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
    
    logger.info(f"Loaded {len(samples)} samples from {sample_file}")
    
    # Apply sample limit if specified
    if sample_limit is not None:
        samples = samples[:sample_limit]
        logger.info(f"Limited to {len(samples)} samples")
    
    # Define output path
    output_path = os.path.join(results_dir, f"gemini_flash_{task_type}_results.json")
    
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
    for i, sample in enumerate(tqdm(samples_to_process, desc=f"Processing {task_type} samples")):
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
        
        # Add a small delay between requests to avoid rate limiting
        time.sleep(2)
    
    logger.info(f"Evaluation completed for {task_type}. Results saved to {output_path}")


def main():
    """Main function to run Gemini-2.5-Flash evaluation on dataset samples."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Gemini-2.5-Flash evaluation on code samples")
    parser.add_argument("--samples_dir", type=str, default="dataset_samples",
                        help="Directory containing dataset samples")
    parser.add_argument("--results_dir", type=str, default="gemini_flash_results",
                        help="Directory to save results")
    parser.add_argument("--task", type=str, choices=["code_generation", "code_generation_bcb_hard", "code_generation_bcb_full", "issue_resolution", "code_translation", "all"],
                        default="code_generation_bcb_hard", help="Task type to evaluate")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Maximum number of samples to process (None for all)")
    parser.add_argument("--skip", type=int, default=None,
                        help="Number of samples to skip")
    
    args = parser.parse_args()
    
    # Run evaluation for specified task(s)
    if args.task == "all":
        tasks_to_run = ["code_generation", "code_generation_bcb_hard", "code_generation_bcb_full", "issue_resolution", "code_translation"]
    else:
        tasks_to_run = [args.task]
    
    for task_type in tasks_to_run:
        logger.info(f"Starting evaluation for {task_type}")
        run_evaluation(
            task_type,
            args.samples_dir,
            args.results_dir,
            args.sample_limit,
            args.skip
        )
    
    logger.info("All evaluations completed")


def run_single_sample():
    """Run a single sample test to verify the API connection."""
    # Get API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    # Initialize OpenAI client with OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Set model parameters
    model_name = "google/gemini-2.5-flash-preview:thinking"
    site_url = os.environ.get("SITE_URL", "https://github.com/user/llm-reasoning")
    site_name = os.environ.get("SITE_NAME", "LLM-Reasoning-Project")
    
    # Simple test prompt
    prompt = "Write a function to calculate the factorial of a number."
    
    # Call the API
    print("Sending request to OpenRouter API...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        extra_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name
        },
        extra_body={},
        stream=False
    )
    
    # Print response
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
    # Uncomment to run a single sample test
    # run_single_sample()
