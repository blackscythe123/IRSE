#!/usr/bin/env python3
"""
HumanEval Silver Dataset Creator - Step 2
Create useful/not useful labels for code+comment pairs
"""

import pandas as pd
import json
import time
import openai
from typing import Dict, List
from tqdm import tqdm
import argparse
from pathlib import Path

class HumanEvalLabeler:
    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-exp:free"):
        """Initialize the labeler."""
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        print(f"🤖 Using model: {model}")
    
    def create_labeling_batch(self, entries_batch: List[Dict]) -> List[Dict]:
        """Create useful/not useful labels for a batch of code+comment pairs."""
        
        # Create batch prompt
        batch_prompt = """Evaluate these code solutions with their comments. For each combination, determine if it's "useful" or "not useful" for learning/understanding.

A combination is "useful" if:
- The comment accurately describes the code
- The code appears to work correctly  
- Together they help understand the solution approach

A combination is "not useful" if:
- The comment is misleading, wrong, or generic
- The code has obvious bugs or errors
- The combination is confusing rather than helpful

Format your response as a JSON array:
[
  {"solution_id": "0_1", "label": "useful"},
  {"solution_id": "0_2", "label": "not useful"},
  ...
]

Code+Comment pairs to evaluate:
"""
        
        for i, entry in enumerate(entries_batch):
            batch_prompt += f"\n{i+1}. Solution ID: {entry['full_id']}\n"
            batch_prompt += f"Comment: {entry['comment']}\n"
            batch_prompt += f"Code:\n{entry['code']}\n"
            batch_prompt += "---\n"
        
        batch_prompt += "\nRespond with ONLY the JSON array containing solution_id and label (useful/not useful)."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.1,
                max_tokens=2000,
                timeout=60
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                labels_data = json.loads(content)
                
                # Create results with labels
                results = []
                for entry in entries_batch:
                    # Find matching label
                    label = "not useful"  # Default fallback
                    for label_data in labels_data:
                        if label_data.get('solution_id') == entry['full_id']:
                            parsed_label = label_data.get('label', '').lower().strip()
                            if 'useful' in parsed_label and 'not useful' not in parsed_label:
                                label = "useful"
                            elif 'not useful' in parsed_label:
                                label = "not useful"
                            break
                    
                    # Format for silver dataset
                    results.append({
                        'Comments': entry['comment'],
                        'Surrounding Code Context': entry['code'],
                        'Class': label,
                        # Metadata
                        'task_id': entry['task_id'],
                        'solution_id': entry['solution_id'],
                        'full_id': entry['full_id'],
                        'function_signature': entry['function_signature'],
                        'labeling_success': True
                    })
                
                return results
                
            except json.JSONDecodeError as e:
                print(f"    ⚠️ JSON parsing failed: {e}")
                # Fallback: assign default labels
                results = []
                for entry in entries_batch:
                    results.append({
                        'Comments': entry['comment'],
                        'Surrounding Code Context': entry['code'],
                        'Class': 'not useful',  # Conservative default
                        'task_id': entry['task_id'],
                        'solution_id': entry['solution_id'],
                        'full_id': entry['full_id'],
                        'function_signature': entry['function_signature'],
                        'labeling_success': False
                    })
                return results
            
        except Exception as e:
            print(f"    ⚠️ Batch labeling failed: {str(e)}")
            
            # Handle rate limits
            if "429" in str(e):
                print(f"    ⏳ Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                return self.create_labeling_batch(entries_batch)  # Retry
            
            # Fallback: assign default labels
            results = []
            for entry in entries_batch:
                results.append({
                    'Comments': entry['comment'],
                    'Surrounding Code Context': entry['code'],
                    'Class': 'not useful',  # Conservative default
                    'task_id': entry['task_id'],
                    'solution_id': entry['solution_id'],
                    'full_id': entry['full_id'],
                    'function_signature': entry['function_signature'],
                    'labeling_success': False,
                    'error': str(e)
                })
            return results
    
    def process_all_entries(self, entries: List[Dict], batch_size: int = 30) -> List[Dict]:
        """Process all entries in batches to create labels."""
        all_results = []
        total_batches = (len(entries) + batch_size - 1) // batch_size
        
        print(f"🔄 Processing {len(entries)} entries in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"\n🏷️  Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Create labels for this batch
            batch_results = self.create_labeling_batch(batch)
            all_results.extend(batch_results)
            
            # Save intermediate results
            if batch_results:
                intermediate_df = pd.DataFrame(batch_results)
                intermediate_file = f"data/labels_batch_{batch_num}.csv"
                intermediate_df.to_csv(intermediate_file, index=False)
                print(f"  💾 Saved batch {batch_num} to {intermediate_file}")
                
                # Show label distribution
                useful_count = sum(1 for r in batch_results if r['Class'] == 'useful')
                not_useful_count = len(batch_results) - useful_count
                success_count = sum(1 for r in batch_results if r['labeling_success'])
                
                print(f"  📊 Labels: {useful_count} useful, {not_useful_count} not useful")
                print(f"  ✅ Success rate: {success_count}/{len(batch_results)} ({success_count/len(batch_results)*100:.1f}%)")
            
            # Delay between batches
            if i + batch_size < len(entries):
                print(f"  ⏳ Waiting 15 seconds before next batch...")
                time.sleep(15)
        
        return all_results
    
    def create_silver_dataset(self, comments_file: str, output_file: str, batch_size: int = 30):
        """Create final silver dataset with useful/not useful labels."""
        print(f"🚀 Creating HumanEval Silver Dataset")
        print(f"📁 Input: {comments_file}")
        print(f"💾 Output: {output_file}")
        print(f"📦 Batch size: {batch_size}")
        
        # Load comments dataset
        comments_df = pd.read_csv(comments_file)
        print(f"📋 Loaded {len(comments_df)} code+comment pairs")
        
        # Convert to list of dicts
        entries = comments_df.to_dict('records')
        
        # Process all entries
        results = self.process_all_entries(entries, batch_size)
        
        if results:
            # Save final silver dataset
            silver_df = pd.DataFrame(results)
            
            # Clean version (just the required columns)
            clean_df = silver_df[['Comments', 'Surrounding Code Context', 'Class']].copy()
            clean_df.to_csv(output_file, index=False)
            
            # Detailed version with metadata
            detailed_file = output_file.replace('.csv', '_detailed.csv')
            silver_df.to_csv(detailed_file, index=False)
            
            print(f"\n✅ Silver dataset saved to: {output_file}")
            print(f"📊 Detailed dataset saved to: {detailed_file}")
            
            # Print final summary
            class_counts = clean_df['Class'].value_counts()
            success_count = sum(1 for r in results if r['labeling_success'])
            
            print(f"\n📈 Final Summary:")
            print(f"  Total items: {len(clean_df)}")
            print(f"  Successfully labeled: {success_count} ({success_count/len(results)*100:.1f}%)")
            for label, count in class_counts.items():
                percentage = (count / len(clean_df)) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
            
            return clean_df
        else:
            print("❌ No results generated!")
            return None

def main():
    parser = argparse.ArgumentParser(description="Create Silver Dataset from Comments")
    parser.add_argument("--input", "-i",
                       default="data/humaneval_with_comments.csv",
                       help="Input CSV file with comments")
    parser.add_argument("--output", "-o",
                       default="data/humaneval_silver_dataset_final.csv",
                       help="Output silver dataset CSV file")
    parser.add_argument("--batch-size", "-b",
                       type=int, default=30,
                       help="Batch size for labeling")
    parser.add_argument("--model",
                       default="google/gemini-2.0-flash-exp:free",
                       choices=["x-ai/grok-4-fast:free", "google/gemini-2.0-flash-exp:free"],
                       help="Model to use for labeling")
    
    args = parser.parse_args()
    
    # API key
    api_key = "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"
    
    # Create labeler
    labeler = HumanEvalLabeler(api_key=api_key, model=args.model)
    
    # Create silver dataset
    result = labeler.create_silver_dataset(
        comments_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    if result is not None:
        print(f"\n🎉 Success! Created silver dataset with {len(result)} labeled examples")
    else:
        print("\n❌ Failed to create silver dataset")

if __name__ == "__main__":
    main()