# CodeBERT + Domain-Adaptive MLM for Source Code Comment Classification

This repository contains the LaTeX source, figures, and tables for the research paper on **enhancing CodeBERT representations using domain-adaptive masked language modeling (MLM)** for source code comment classification. The study uses the IRSE/FIRE dataset and explores the impact of combining original C code data with Python-derived silver-standard examples generated using **x-ai/grok-4-fast**.

---

## 📄 Paper Overview

**Title:** Improving Source Code Comment Relevance with Domain-Adaptive CodeBERT  

**Authors:**  
- L. Simiyon Vinscent Samuel  
- Sundareswaran Rajaguru  
- Vijayaraaghavan K. S.  
- Dr. A. Vijayalakshmi (Supervisor)

**Affiliation:**  
Department of Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering, Chennai, India

---

## 🧪 Key Highlights

- **Datasets:**  
  - Original IRSE/FIRE C code dataset (11,452 pairs, 8,326 cleaned training samples)  
  - Rule-based synthetic comments (500 samples)  
  - Silver-standard Python examples (1,640 samples) generated from `humaneval.jsonl` via `x-ai/grok-4-fast`  

- **Training Approach:**  
  - CodeBERT base MLM continued on combined corpus  
  - 15% mask probability, batch size 16, learning rate 5e-5, up to 7 epochs  
  - Fine-tuning with a classification head for relevance prediction  

- **Baselines:**  
  - TF–IDF features with Logistic Regression, Linear SVM, Multinomial NB, Random Forest  
  - 10-fold cross-validation for hyperparameter tuning  

- **Evaluation Metrics:** Accuracy, Precision, Recall, Weighted F1  
- **Configurations:**  
  1. Train Original → Test Original  
  2. Train Combined → Test Combined  
  3. Original → Combined  
  4. Combined → Original  

- **Findings:**  
  - Domain-adaptive MLM significantly improves CodeBERT representations.  
  - Cross-evaluation shows robust transfer even with Python-derived silver data.  
  - Silver-standard augmentation increases vocabulary and diversity but introduces minor noise.  

