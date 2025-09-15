#!/usr/bin/env python3
"""
Run DeepSeek-R1 model on various code-related tasks.
"""
import os
import json
import time
import logging
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deepseek_r1_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DeepSeekRunner:
    """Run DeepSeek-R1 model for code-related tasks."""
    
    def __init__(self):
        """Initialize DeepSeek runner."""
        # Get API key from environment variable
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        
        # Initialize client
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    def generate_completion(self, prompt):
        """Generate completion using DeepSeek-R1.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict with content, reasoning_content, error, and duration
        """
        start_time = time.time()
        
        try:
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=4000
            )
            
            # Extract content and reasoning content
            content = response.choices[0].message.content
            reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, 'reasoning_content') else ""
            
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
            
            return {
                "content": "",
                "reasoning_content": "",
                "error": f"Error code: {e.status_code if hasattr(e, 'status_code') else 'unknown'} - {str(e)}",
                "duration": duration
            }

class PromptFormatter:
    """Format prompts for different task types."""
    
    def __init__(self, task_type: str):
        """Initialize prompt formatter.
        
        Args:
            task_type: Type of task ('code_generation', 'issue_resolution', or 'code_translation')
        """
        self.task_type = task_type
    
    def format_prompt(self, sample: Dict) -> Dict:
        """Format prompt based on task type.
        
        Args:
            sample: Sample data
            
        Returns:
            Dict with formatted prompt
        """
        if self.task_type == "code_generation" or self.task_type == "code_generation_bcb_simple":
            return self.format_code_generation_prompt(sample)
        elif self.task_type == "issue_resolution":
            # Currently not using issue resolution
            # return self.format_issue_resolution_prompt(sample)
            raise ValueError(f"Issue resolution task type is currently commented out")
        elif self.task_type == "code_translation":
            # Currently not using code translation
            # return self.format_code_translation_prompt(sample)
            raise ValueError(f"Code translation task type is currently commented out")
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    class PromptFormatter:
        """Format prompts for different task types."""
    
    def __init__(self, task_type: str):
        """Initialize prompt formatter.
        
        Args:
            task_type: Type of task ('full' or 'hard')
        """
        self.task_type = task_type
    
    def format_prompt(self, sample: Dict) -> Dict:
        """Format prompt based on task type.
        
        Args:
            sample: Sample data
            
        Returns:
            Dict with formatted prompt
        """
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
[Provide your complete code implementation here. Ensure it is functional, efficient, and addresses all requirements.]
"""
        
        return {"prompt": prompt}
    
    # def format_issue_resolution_prompt(self, sample):
    #     """Format prompt for issue resolution task."""
    #     task_id = sample.get("task_id", "Unknown")
    #     repo = sample.get("repo", "Unknown")
    #     issue_title = sample.get("issue_title", "No issue title provided.")
    #     issue_body = sample.get("issue_body", "No issue description provided.")
    #     
    #     prompt = f"""# Issue Resolution Task (ID: {task_id})
    # 
    # ## Repository
    # {repo}
    # 
    # ## Issue Title
    # {issue_title}
    # 
    # ## Issue Description
    # {issue_body}
    # 
    # Imagine you are an expert software developer with years of experience in solving complex issues.
    # Analyze this issue step by step. First understand the problem, identify potential causes, and then suggest a solution. 
    # Explain your reasoning process thoroughly."""
    #     
    #     return {"prompt": prompt}
    
    # def format_code_translation_prompt(self, sample):
    #     """Format prompt for code translation task."""
    #     task_id = sample.get("task_id", "Unknown")
    #     source_language = sample.get("source_language", "Unknown")
    #     target_language = sample.get("target_language", "Unknown")
    #     source_code = sample.get("source_code", "No source code provided.")
    #     
    #     prompt = f"""# Code Translation Task (ID: {task_id})
    # 
    # ## Source Language
    # {source_language}
    # 
    # ## Target Language
    # {target_language}
    # 
    # ## Source Code
    # ```{source_language}
    # {source_code}
    # ```
    # 
    # Imagine you are an expert software developer with years of experience in multiple programming languages.
    # Translate this code from {source_language} to {target_language}. First analyze the code to understand its functionality, 
    # then plan how to implement it in the target language, and finally provide the translation. 
    # Explain your reasoning process thoroughly."""
    #     
    #     return {"prompt": prompt}

def run_evaluation(samples_path: str, task_type: str, output_path: str, sample_limit: int = None):
    """Run evaluation on samples using DeepSeek-R1.
    
    Args:
        samples_path: Path to samples file
        task_type: Type of task
        output_path: Path to save results
        sample_limit: Maximum number of samples to process (None for all)
    """
    # Load samples
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    # Apply sample limit if specified
    if sample_limit is not None and sample_limit < len(samples):
        samples = samples[:sample_limit]
    
    logger.info(f"Running evaluation on {len(samples)} samples")
    
    # Initialize components
    runner = DeepSeekRunner()
    formatter = PromptFormatter(task_type)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process each sample
    results = []
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {task_type} samples")):
        logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.get('task_id', f'Sample_{i}')}")
        
        # Format prompt
        formatted = formatter.format_prompt(sample)
        prompt = formatted.get("prompt", "")
        
        # Generate completion
        completion = runner.generate_completion(prompt=prompt)
        
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
        
        logger.info(f"Completed sample {i+1}/{len(samples)}")
    
    logger.info(f"Evaluation completed for {task_type}. Results saved to {output_path}")

def main():
    """Main function to run DeepSeek-R1 evaluation on dataset samples."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepSeek-R1 evaluation on code samples")
    parser.add_argument("--samples_dir", type=str, default="dataset_samples",
                        help="Directory containing dataset samples")
    parser.add_argument("--results_dir", type=str, default="deepseek_results",
                        help="Directory to save results")
    parser.add_argument("--task", type=str, choices=["code_generation", "code_generation_bcb_simple", "code_generation_bcb_hard", "issue_resolution", "code_translation", "all"],
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
        "code_generation_bcb_hard": {"pattern": "bigcodebench_v0.1.0_hf_instruct_samples.jsonl"},
        # Currently not using these task types
        # "issue_resolution": {"pattern": "swe-bench*_samples.jsonl"},
        # "code_translation": {"pattern": "codetransocean_samples.jsonl"}
    }
    
    # Get actual sample files
    import glob
    sample_files = {}
    for task_type, config in task_configs.items():
        pattern = os.path.join(args.samples_dir, config["pattern"])
        matching_files = glob.glob(pattern)
        if matching_files:
            sample_files[task_type] = matching_files[0]  # Use the first matching file
    
    # Run evaluation for specified task(s)
    if args.task == "all":
        tasks_to_run = list(task_configs.keys())
    else:
        tasks_to_run = [args.task]
    
    for task_type in tasks_to_run:
        if task_type in sample_files:
            samples_path = sample_files[task_type]
            # output_path = os.path.join(args.results_dir, f"{task_type}_results.json")
            output_path = os.path.join(args.results_dir, f"deepseek_{task_type}_results.json")
            
            logger.info(f"Starting evaluation for {task_type} using {samples_path}")
            run_evaluation(samples_path, task_type, output_path, args.sample_limit)
        else:
            logger.warning(f"No sample file found for task type: {task_type}")
    
    logger.info("All evaluations completed")

if __name__ == "__main__":
    main()
