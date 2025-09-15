#!/usr/bin/env python3
"""
Run OpenAI o3-mini model on BigCodeBench tasks to evaluate reasoning capabilities.
"""
import os
import json
import time
import logging
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("o3_mini_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class OpenAIReasoningRunner:
    """Run OpenAI o3-mini model for code-related tasks with reasoning."""
    
    # def __init__(self, reasoning_effort="high"):
    def __init__(self, reasoning_effort="low"):
        """Initialize OpenAI reasoning runner.
        
        Args:
            reasoning_effort: Effort level for reasoning ("low", "medium", or "high")
        """
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize client
        self.client = OpenAI(api_key=api_key)
        self.reasoning_effort = reasoning_effort
    
    def generate_completion(self, prompt):
        """Generate completion using OpenAI o3-mini with reasoning.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict with content, reasoning_content, usage_stats, error, and duration
        """
        start_time = time.time()
        
        try:
            # Call OpenAI API with reasoning (non-streaming)
            response = self.client.responses.create(
                model="o3-mini",
                reasoning={"effort": self.reasoning_effort},
                input=[
                    {"role": "user", "content": prompt}
                ],
                max_output_tokens=8000  # Adjust based on your needs
            )
            
            # Extract content
            content = response.output_text
            
            # Extract reasoning content using multiple methods
            reasoning_content = self._extract_reasoning_content(response, content)
            
            # Get token usage stats
            usage_stats = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens if hasattr(response.usage.output_tokens_details, 'reasoning_tokens') else 0,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate duration
            duration = time.time() - start_time
            
            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "usage_stats": usage_stats,
                "error": None,
                "duration": duration,
                "response_id": response.id  # Save response ID for potential future use
            }
            
        except Exception as e:
            # Handle errors
            logger.error(f"Error generating completion: {str(e)}")
            duration = time.time() - start_time
            
            return {
                "content": "",
                "reasoning_content": "",
                "usage_stats": {},
                "error": f"Error - {str(e)}",
                "duration": duration,
                "response_id": None
            }
    
    def _extract_reasoning_content(self, response, content):
        """Extract reasoning content using multiple methods.
        
        Args:
            response: The API response object
            content: The text content from the response
            
        Returns:
            String containing the extracted reasoning content
        """
        # First try to extract from API response structure (if available)
        try:
            if hasattr(response, 'items'):
                for item in response.items:
                    if item.type == "reasoning":
                        if hasattr(item, 'value') and item.value:
                            return item.value
                        elif hasattr(item, 'summary') and item.summary:
                            return "\n\n".join([summary.value for summary in item.summary])
        except Exception as e:
            logger.warning(f"Failed to extract reasoning from response items: {str(e)}")
        
        # If API structure extraction failed, use text parsing approach from MedRBench
        return self._get_reasoning_content(content)
    
    def _get_reasoning_content(self, content):
        """Extract reasoning content from text using pattern matching.
        
        This method is adapted from the MedRBench project's get_reasoning_content function.
        
        Args:
            content: The text content from the response
            
        Returns:
            String containing the extracted reasoning content
        """
        if not content:
            return ""
            
        # Clean up the content
        clean_content = content.replace('```', '').strip()
        
        # Try to extract reasoning based on section headers
        reasoning_content = ""
        
        # Check for various section headers that might indicate reasoning content
        reasoning_markers = [
            '### Reasoning Process:', 
            '### Reasoning:', 
            '**Reasoning:**', 
            '### Chain of Thought:', 
            'Explanation:', 
            'Step-by-Step Reasoning',
            'Analysis:',
            'Approach:',
            'Design:',
            'Algorithm:'
        ]
        
        # Find the first matching marker
        found_marker = None
        for marker in reasoning_markers:
            if marker in clean_content:
                found_marker = marker
                break
        
        if found_marker:
            # Extract content after the marker
            parts = clean_content.split(found_marker)
            if len(parts) > 1:
                reasoning_content = parts[1].strip()
                
                # Try to find where the reasoning ends
                end_markers = [
                    '### Solution:', 
                    '### Conclusion:', 
                    '### Answer:', 
                    'Final Code Solution', 
                    'Final Code:',
                    'Conclusion:',
                    'Solution:',
                    '```python',
                    '```'
                ]
                
                for end_marker in end_markers:
                    if end_marker in reasoning_content:
                        reasoning_content = reasoning_content.split(end_marker)[0].strip()
                        break
        
        # If no explicit markers found, try to extract based on content structure
        if not reasoning_content:
            # Look for step patterns
            if re.search(r'<step\s+\d+>', clean_content) or re.search(r'Step\s+\d+[:.]\s+', clean_content):
                # Find the end of the reasoning section
                for marker in ['import ', 'def ', 'class ', '```']:
                    if marker in clean_content:
                        reasoning_content = clean_content.split(marker)[0].strip()
                        break
            
            # If still nothing, try to extract the first part of the content before code
            if not reasoning_content:
                code_indicators = ['import ', 'def ', 'class ', '```']
                for indicator in code_indicators:
                    if indicator in clean_content:
                        reasoning_content = clean_content.split(indicator)[0].strip()
                        break
                
                # If still nothing, use the whole content as reasoning
                if not reasoning_content:
                    reasoning_content = clean_content
        
        return reasoning_content.strip()
    
    def _split_reasoning_steps(self, reasoning_content, max_steps=10):
        """Split reasoning content into steps.
        
        This method is adapted from the MedRBench project's split_reasoning function.
        
        Args:
            reasoning_content: The extracted reasoning content
            max_steps: Maximum number of steps to return
            
        Returns:
            List of reasoning steps
        """
        try:
            # Extract steps using regex pattern <step 1> <step 2>
            pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
            matches = re.findall(pattern, reasoning_content, re.DOTALL)
            reasoning_steps = [step_content.strip() for _, step_content in matches]
            
            # If no steps found with <step N> pattern, try numbered list pattern
            if len(reasoning_steps) == 0:
                pattern = r"(?:^|\n)(?:Step\s+)?(\d+)[:.]\s+(.*?)(?=(?:^|\n)(?:Step\s+)?(?:\d+)[:.]\s+|$)"
                matches = re.findall(pattern, reasoning_content, re.DOTALL)
                reasoning_steps = [step_content.strip() for _, step_content in matches]
            
            # If still no steps found, try splitting by paragraphs
            if len(reasoning_steps) == 0:
                if '\n\n' in reasoning_content:
                    reasoning_steps = reasoning_content.split('\n\n')
                else:
                    reasoning_steps = reasoning_content.split('\n')
            
            return reasoning_steps[:max_steps]  # Limit to max_steps if specified
            
        except Exception as e:
            logger.warning(f"Failed to split reasoning steps: {str(e)}")
            return [reasoning_content] if reasoning_content else []

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

# def run_evaluation(samples_path: str, task_type: str, output_path: str, reasoning_effort: str = "medium", sample_limit: int = None):
# def run_evaluation(samples_path: str, task_type: str, output_path: str, reasoning_effort: str = "medium", sample_limit: int = None, skip: int = 0):
# def run_evaluation(samples_path: str, task_type: str, output_path: str, reasoning_effort: str = "high", sample_limit: int = None, skip: int = 0):
def run_evaluation(samples_path: str, task_type: str, output_path: str, reasoning_effort: str = "low", sample_limit: int = None, skip: int = 0):
    """Run evaluation on samples using OpenAI o3-mini.
    
    Args:
        samples_path: Path to samples file
        task_type: Type of task ('full' or 'hard')
        output_path: Path to save results
        reasoning_effort: Effort level for reasoning ("low", "medium", or "high")
        sample_limit: Maximum number of samples to process (None for all)
        skip: Number of samples to skip from the beginning
    """
    # Load samples
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    # Skip samples if specified
    if skip > 0:
        if skip >= len(samples):
            logger.warning(f"Skip count ({skip}) is greater than or equal to the number of samples ({len(samples)}). No samples to process.")
            return
        samples = samples[skip:]
    
    # Apply sample limit if specified
    if sample_limit is not None and sample_limit < len(samples):
        samples = samples[:sample_limit]

    # logger.info(f"Running evaluation on {len(samples)} samples from {task_type} dataset")
    logger.info(f"Running evaluation on {len(samples)} samples from {task_type} dataset (skipped {skip} samples)")
    
    # Initialize components
    runner = OpenAIReasoningRunner(reasoning_effort=reasoning_effort)
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
            "usage_stats": completion.get("usage_stats", {}),
            "error": completion.get("error"),
            "duration": completion.get("duration"),
            "response_id": completion.get("response_id"),
            "original_sample": sample
        }
        
        # Add to results
        results.append(result)
        
        # Save results after each sample (in case of interruption)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed sample {i+1}/{len(samples)}")
        
        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)
    
    logger.info(f"Evaluation completed for {task_type}. Results saved to {output_path}")

def main():
    """Main function to run OpenAI o3-mini evaluation on BigCodeBench samples."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenAI o3-mini evaluation on BigCodeBench samples")
    parser.add_argument("--samples_dir", type=str, default="dataset_samples",
                        help="Directory containing dataset samples")
    parser.add_argument("--results_dir", type=str, default="o3_mini_results",
                        help="Directory to save results")
    parser.add_argument("--task", type=str, choices=["full", "hard", "both"],
                        default="both", help="Task type to evaluate")
    # parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"],
    #                     default="medium", help="Effort level for reasoning")
    # parser.add_argument("--reasoning_effort", type=str, choices=["high"],
    #                     default="high", help="Effort level for reasoning")
    parser.add_argument("--reasoning_effort", type=str, choices=["low"],
                        default="low", help="Effort level for reasoning")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Maximum number of samples to process per task (None for all)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Number of samples to skip from the beginning of the dataset")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define task configurations with expected sample file patterns
    task_configs = {
        "full": {"pattern": "bigcodebench_v0.1.0_hf_instruct_full_samples.jsonl"},
        "hard": {"pattern": "bigcodebench_v0.1.0_hf_instruct_samples.jsonl"}
    }
    
    # Get actual sample files
    import glob
    sample_files = {}
    for task_type, config in task_configs.items():
        pattern = os.path.join(args.samples_dir, config["pattern"])
        matching_files = glob.glob(pattern)
        if matching_files:
            sample_files[task_type] = matching_files[0]  # Use the first matching file
        else:
            # Try alternative pattern
            alt_pattern = os.path.join(args.samples_dir, f"*{task_type}*_samples.jsonl")
            matching_files = glob.glob(alt_pattern)
            if matching_files:
                sample_files[task_type] = matching_files[0]
    
    # Run evaluation for specified task(s)
    if args.task == "both":
        tasks_to_run = ["full", "hard"]
    else:
        tasks_to_run = [args.task]
    
    for task_type in tasks_to_run:
        if task_type in sample_files:
            samples_path = sample_files[task_type]
            # output_path = os.path.join(args.results_dir, f"o3_mini_{task_type}_results.json")
            # output_path = os.path.join(args.results_dir, f"o3_mini_high_{task_type}_results.json")
            output_path = os.path.join(args.results_dir, f"o3_mini_low_{task_type}_results.json")

            logger.info(f"Starting evaluation for {task_type} using {samples_path}")
            run_evaluation(
                samples_path, 
                task_type, 
                output_path, 
                reasoning_effort=args.reasoning_effort, 
                sample_limit=args.sample_limit,
                skip=args.skip
            )
        else:
            logger.warning(f"No sample file found for task type: {task_type}")
            logger.warning(f"Please ensure the samples file for {task_type} follows the pattern: bigcodebench_{task_type}_samples.jsonl")
    
    logger.info("All evaluations completed")

if __name__ == "__main__":
    main()