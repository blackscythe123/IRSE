#!/usr/bin/env python3
"""
HumanEval Silver Dataset Summary
"""

import pandas as pd

def analyze_dataset():
    # Load the final dataset
    df = pd.read_csv('data/humaneval_silver_dataset_final.csv')
    
    print("🎉 HumanEval Silver Dataset Analysis")
    print("=" * 50)
    
    print(f"📊 Dataset Size: {len(df):,} examples")
    print(f"📝 Average comment length: {df['Comments'].str.len().mean():.1f} characters")
    print(f"💻 Average code length: {df['Surrounding Code Context'].str.len().mean():.1f} characters")
    
    print("\n🏷️ Label Distribution:")
    label_counts = df['Class'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print("\n📈 Sample Examples:")
    print("-" * 30)
    
    # Show a few examples of each class
    useful_examples = df[df['Class'] == 'useful'].head(2)
    not_useful_examples = df[df['Class'] == 'not useful'].head(2)
    
    print("\n✅ USEFUL Examples:")
    for i, row in useful_examples.iterrows():
        print(f"\nExample {i+1}:")
        print(f"Comment: {row['Comments'][:100]}...")
        print(f"Code: {row['Surrounding Code Context'][:100]}...")
    
    print("\n❌ NOT USEFUL Examples:")
    for i, row in not_useful_examples.iterrows():
        print(f"\nExample {i+1}:")
        print(f"Comment: {row['Comments'][:100]}...")
        print(f"Code: {row['Surrounding Code Context'][:100]}...")
    
    print(f"\n✨ Successfully created silver dataset with balanced labels!")
    print(f"📁 File: data/humaneval_silver_dataset_final.csv")
    print(f"💾 Size: {len(df):,} labeled code+comment pairs")

if __name__ == "__main__":
    analyze_dataset()