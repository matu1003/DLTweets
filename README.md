# 🐦 Tweet Influence Classification  
Predicting whether a tweet was posted by an **influencer** or by a **regular user**.

---

## 🔍 Project Overview

This project explores multiple modelling strategies to classify tweets into two categories:

- **1 — Influencer**
- **0 — Regular user**

We gradually build the solution in three stages:

1. **Text-only models** (`baseline.ipynb`)
2. **Metadata-only models** (`withmeta.ipynb`)
3. **Hybrid transformer + metadata models** (`metabert.ipynb`)

The repository contains experiments with classical ML models (TF-IDF, linear/logistic regression, decision trees), transformer encoders (BERT, CamemBERT), feature engineering, and a fused architecture where textual embeddings are combined with structured metadata.

---

## 📂 Repository Structure
```
├── baseline.ipynb # Text-only baseline experiments 
├── withmeta.ipynb # Metadata exploration + ML models
├── metabert.ipynb # Fusion of BERT/CamemBERT embeddings with metadata
├── feature_analysis_pablo.ipynb
├── test1.ipynb
├── test2_pablo.ipynb
├── Camemberth.ipynb
├── MetaBerth.ipynb
├── vocab.py # Custom vocabulary helper 
├── Data/
│ ├── train.jsonl
│ ├── kaggle_test.jsonl
│ └── ...
├── README.md
└── requirements.txt
```
---

# 1️⃣ Baseline — Text-Only Models (`baseline.ipynb`)

This notebook establishes a foundation by using **only the tweet text**.

### 🛠 Methods
- Classical ML text pipelines:
  - TF–IDF vectorization
  - Logistic Regression
  - Linear models
  - Decision Trees
- Transformer-based encoders:
  - **BERT-base**
  - **CamemBERT**

### 🎯 Key Result
- Best accuracy: **~70%** using transformer embeddings.
- Conclusion: the text signal is helpful, but not sufficient alone.

---

# 2️⃣ Metadata-Only Models (`withmeta.ipynb`)

Tweets come with a large set of structured features (tweet metadata + user metadata).  
This notebook focuses on leveraging them effectively.

### 🧩 Work done
- Metadata cleaning & selection  
- One-hot encoding, scaling, type conversions  
- Feature importance analysis  
- Experiments with:
  - Logistic Regression
  - Random Forests
  - Gradient Boosting
  - XGBoost / LightGBM

### 🎯 Findings
- Metadata is **highly predictive**.
- Metadata-only models frequently outperform text-only baselines.
- Provides a strong foundation for hybrid modelling.

---

# 3️⃣ MetaBERT — Fusion of Text (BERT) + Metadata (`metabert.ipynb`)

This notebook builds a **hybrid architecture** combining:

- A **transformer encoder** (BERT or CamemBERT) for the tweet text
- A **dense network** for metadata features
- A final **concatenation layer** followed by a classifier

### 🧠 Model Architecture
Text → BERT/CamemBERT → CLS embedding
Metadata → Dense layers
Final representation → Concat([CLS], metadata_vector) → Linear → Output


### 💡 Tools and Libraries
- HuggingFace `transformers`
- PyTorch
- scikit-learn
- numpy / pandas

### 🎯 Results
- Clear improvement over text-only and metadata-only approaches.
- More stable and better generalization.
- This hybrid model represents the **final recommended architecture**.

---


# ✨ Authors

Work conducted by Mathurin PETIT and Pablo POULENARD, building progressively toward a robust influencer detection model.
---
