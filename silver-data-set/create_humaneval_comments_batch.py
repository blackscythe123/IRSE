#!/usr/bin/env python3
"""
HumanEval Comment Generator - Batch Processing
Step 1: Generate comments for all 1640 code solutions using high-context models
"""

import pandas as pd
import json
import time
import openai
from typing import Dict, List
from tqdm import tqdm
import argparse
from pathlib import Path

class HumanEvalCommentGenerator:
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast:free"):
        """Initialize the comment generator."""
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        print(f"🤖 Using model: {model}")
    
    def load_humaneval_data(self, file_path: str) -> List[Dict]:
        """Load HumanEval data from JSONL file."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line)
                    data.append(task_data)
        return data
    
    def extract_all_solutions(self, tasks: List[Dict]) -> List[Dict]:
        """Extract all code solutions from all tasks."""
        all_solutions = []
        
        for task_data in tasks:
            task_id = task_data.get('task_id', 'unknown')
            prompt = task_data.get('prompt', '')
            
            # Extract function signature from prompt
            lines = prompt.strip().split('\n')
            function_signature = None
            for line in lines:
                if line.strip().startswith('def '):
                    function_signature = line.strip()
                    break
            
            # Extract all code solutions
            for i in range(1, 11):  # code_1 to code_10
                code_key = f'code_{i}'
                if code_key in task_data:
                    all_solutions.append({
                        'task_id': task_id,
                        'solution_id': i,
                        'function_signature': function_signature,
                        'prompt': prompt,
                        'code': task_data[code_key],
                        'full_id': f"{task_id}_{i}"
                    })
        
        return all_solutions
    
    def generate_comments_batch(self, solutions_batch: List[Dict]) -> List[Dict]:
        """Generate comments for a batch of solutions using high-context model."""
        
        # Create batch prompt
        batch_prompt = """Generate brief, helpful comments for these Python code solutions. For each code solution, write a concise comment (1-2 sentences) that explains what the code does or how it works.

Format your response as a JSON array with this structure:
[
  {"solution_id": "0_1", "comment": "Your comment here"},
  {"solution_id": "0_2", "comment": "Your comment here"},
  ...
]

Code solutions to comment:
"""
        
        for i, solution in enumerate(solutions_batch):
            batch_prompt += f"\n{i+1}. Solution ID: {solution['full_id']}\n"
            batch_prompt += f"Function: {solution['function_signature']}\n"
            batch_prompt += f"Code:\n{solution['code']}\n"
            batch_prompt += "---\n"
        
        batch_prompt += "\nRespond with ONLY the JSON array, no other text."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.3,
                max_tokens=4000,  # Higher for batch processing
                timeout=60  # Longer timeout for larger batches
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                comments_data = json.loads(content)
                
                # Create results with comments
                results = []
                for solution in solutions_batch:
                    # Find matching comment
                    comment = "Generic comment: Code solution for the given function."
                    for comment_data in comments_data:
                        if comment_data.get('solution_id') == solution['full_id']:
                            comment = comment_data.get('comment', comment)
                            break
                    
                    results.append({
                        'task_id': solution['task_id'],
                        'solution_id': solution['solution_id'],
                        'full_id': solution['full_id'],
                        'function_signature': solution['function_signature'],
                        'code': solution['code'],
                        'comment': comment,
                        'success': True
                    })
                
                return results
                
            except json.JSONDecodeError as e:
                print(f"    ⚠️ JSON parsing failed: {e}")
                # Fallback: create generic comments
                results = []
                for solution in solutions_batch:
                    results.append({
                        'task_id': solution['task_id'],
                        'solution_id': solution['solution_id'],
                        'full_id': solution['full_id'],
                        'function_signature': solution['function_signature'],
                        'code': solution['code'],
                        'comment': "Generic comment: Code solution for the given function.",
                        'success': False
                    })
                return results
            
        except Exception as e:
            print(f"    ⚠️ Batch comment generation failed: {str(e)}")
            
            # Handle rate limits
            if "429" in str(e):
                print(f"    ⏳ Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self.generate_comments_batch(solutions_batch)  # Retry
            
            # Fallback: create generic comments
            results = []
            for solution in solutions_batch:
                results.append({
                    'task_id': solution['task_id'],
                    'solution_id': solution['solution_id'],
                    'full_id': solution['full_id'],
                    'function_signature': solution['function_signature'],
                    'code': solution['code'],
                    'comment': f"Error generating comment: {str(e)}",
                    'success': False
                })
            return results
    
    def process_all_solutions(self, solutions: List[Dict], batch_size: int = 50) -> List[Dict]:
        """Process all solutions in batches to generate comments."""
        all_results = []
        total_batches = (len(solutions) + batch_size - 1) // batch_size
        
        print(f"🔄 Processing {len(solutions)} solutions in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(solutions), batch_size):
            batch = solutions[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Generate comments for this batch
            batch_results = self.generate_comments_batch(batch)
            all_results.extend(batch_results)
            
            # Save intermediate results
            if batch_results:
                intermediate_df = pd.DataFrame(batch_results)
                intermediate_file = f"data/comments_batch_{batch_num}.csv"
                intermediate_df.to_csv(intermediate_file, index=False)
                print(f"  💾 Saved batch {batch_num} to {intermediate_file}")
                
                # Show success rate
                success_count = sum(1 for r in batch_results if r['success'])
                print(f"  ✅ Success rate: {success_count}/{len(batch_results)} ({success_count/len(batch_results)*100:.1f}%)")
            
            # Delay between batches (shorter since we're using high-context models)
            if i + batch_size < len(solutions):
                print(f"  ⏳ Waiting 10 seconds before next batch...")
                time.sleep(10)
        
        return all_results
    
    def create_comments_dataset(self, input_file: str, output_file: str, batch_size: int = 50):
        """Create dataset with all code solutions and their comments."""
        print(f"🚀 Creating HumanEval Comments Dataset")
        print(f"📁 Input: {input_file}")
        print(f"💾 Output: {output_file}")
        print(f"📦 Batch size: {batch_size}")
        
        # Load HumanEval data
        tasks = self.load_humaneval_data(input_file)
        print(f"📋 Loaded {len(tasks)} tasks")
        
        # Extract all code solutions
        all_solutions = self.extract_all_solutions(tasks)
        print(f"🔧 Extracted {len(all_solutions)} code solutions")
        
        # Process all solutions
        results = self.process_all_solutions(all_solutions, batch_size)
        
        if results:
            # Save final dataset
            comments_df = pd.DataFrame(results)
            comments_df.to_csv(output_file, index=False)
            
            print(f"\n✅ Comments dataset saved to: {output_file}")
            
            # Print summary
            success_count = sum(1 for r in results if r['success'])
            print(f"\n📈 Final Summary:")
            print(f"  Total solutions: {len(results)}")
            print(f"  Successfully commented: {success_count} ({success_count/len(results)*100:.1f}%)")
            print(f"  Failed: {len(results) - success_count}")
            
            return comments_df
        else:
            print("❌ No results generated!")
            return None

def main():
    parser = argparse.ArgumentParser(description="Generate Comments for HumanEval Dataset")
    parser.add_argument("--input", "-i",
                       default="data/humaneval-codestral-sol.jsonl",
                       help="Input HumanEval JSONL file")
    parser.add_argument("--output", "-o",
                       default="data/humaneval_with_comments.csv",
                       help="Output CSV file with comments")
    parser.add_argument("--batch-size", "-b",
                       type=int, default=50,
                       help="Batch size for processing (higher for high-context models)")
    parser.add_argument("--model",
                       default="x-ai/grok-4-fast:free",
                       choices=["x-ai/grok-4-fast:free", "google/gemini-2.0-flash-exp:free"],
                       help="Model to use for comment generation")
    
    args = parser.parse_args()
    
    # API key
    api_key = "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"
    
    # Create comment generator
    generator = HumanEvalCommentGenerator(api_key=api_key, model=args.model)
    
    # Generate comments dataset
    result = generator.create_comments_dataset(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    if result is not None:
        print(f"\n🎉 Success! Created comments dataset with {len(result)} entries")
        print(f"\nNext step: Run the labeling script to create useful/not useful labels")
    else:
        print("\n❌ Failed to create comments dataset")

if __name__ == "__main__":
    main()