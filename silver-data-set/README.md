# HumanEval Silver Dataset

A comprehensive silver standard dataset for evaluating code+comment quality using the HumanEval benchmark.

## 🎯 Overview

This project successfully created a **silver dataset with 1,640 labeled examples** of code+comment pairs from the HumanEval benchmark, each labeled as "useful" or "not useful" for learning and understanding.

## 📊 Dataset Statistics

- **Total Examples**: 1,640 labeled code+comment pairs
- **Source**: 164 HumanEval tasks × 10 code solutions each
- **Label Distribution**: 
  - Useful: 970 (59.1%)
  - Not useful: 670 (40.9%)
- **Average Comment Length**: 136.4 characters
- **Average Code Length**: 149.7 characters

## 🗂️ Files

### Core Dataset Files
- **`data/humaneval_silver_dataset_final.csv`** - Main silver dataset (clean format)
- **`data/humaneval_silver_dataset_final_detailed.csv`** - Extended version with metadata
- **`data/humaneval-codestral-sol.jsonl`** - Source HumanEval data

### Scripts
- **`create_humaneval_comments_batch.py`** - Generate comments using LLM (Step 1)
- **`create_humaneval_labels_batch.py`** - Create useful/not useful labels (Step 2)
- **`analyze_dataset.py`** - Dataset analysis and statistics

## 🚀 Usage

### Analyze the Dataset
```bash
python analyze_dataset.py
```

### Recreate the Dataset

#### Step 1: Generate Comments
```bash
python create_humaneval_comments_batch.py --model "x-ai/grok-4-fast:free" --batch-size 50
```

#### Step 2: Create Labels
```bash
python create_humaneval_labels_batch.py --model "x-ai/grok-4-fast:free" --batch-size 10
```

## 📋 Data Format

The final silver dataset (`humaneval_silver_dataset_final.csv`) contains:

| Column | Description |
|--------|-------------|
| `Comments` | LLM-generated explanations of the code |
| `Surrounding Code Context` | The actual Python code implementation |
| `Class` | Binary label: "useful" or "not useful" |

### Example Row
```csv
Comments,Surrounding Code Context,Class
"This function sorts the list of numbers and checks if the difference between any two consecutive elements is less than the threshold.","def has_close_elements(numbers, threshold):
    numbers.sort()
    return any(numbers[i+1] - numbers[i] < threshold for i in range(len(numbers) - 1))",useful
```

## 🔧 Technical Details

- **Comment Generation**: Used `x-ai/grok-4-fast:free` model with batch processing (50 solutions per batch)
- **Label Creation**: Used `x-ai/grok-4-fast:free` model with smaller batches (10 items per batch)
- **Success Rates**: 93.9% comment generation, 98.2% labeling success
- **Processing Time**: ~2 hours total with rate limiting
- **API Provider**: OpenRouter with free tier models

## 🎯 Use Cases

This silver dataset is ready for:
- Training ML models to evaluate code+comment quality
- Benchmarking code documentation systems
- Research on code comprehension and explanation
- Fine-tuning language models for code commenting

## 📦 Requirements

```bash
pip install pandas openai tqdm
```