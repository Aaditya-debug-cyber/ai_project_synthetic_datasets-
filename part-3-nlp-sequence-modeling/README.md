# Part 3 – NLP and Sequence Modeling Mini Project

## Overview
End-to-end NLP pipeline for **customer support sentiment classification** (positive / neutral / negative).

## Dataset
`customer_support_text_classification.csv` — 1 500 records with columns:
`ticket_id`, `channel`, `customer_message`, `sentiment_label`, `word_count`, `urgent_flag`

## Tasks Completed

| Task | Description |
|------|-------------|
| 1 | Dataset understanding — EDA, class distribution, sample messages |
| 2 | Text preprocessing — lowercasing, punctuation removal, stopword removal |
| 3 | Vectorisation — Bag of Words, TF-IDF, tokeniser-based integer sequences |
| 4 | Baseline models — Naive Bayes + BoW; Logistic Regression + TF-IDF |
| 5 | LSTM architecture design with layer-by-layer explanation |
| 6 | Reflection on RNNs, LSTMs, attention, and transformers |

## Repository Structure

```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb                  ← main project notebook
├── requirements.txt
├── customer_support_text_classification.csv
└── results/
    ├── model_evaluation.png        ← accuracy/F1 bar chart + confusion matrices
    ├── model_evaluation.csv        ← numeric results
    ├── lstm_architecture.png       ← LSTM layer diagram
    └── sample_predictions.txt      ← 20 example predictions
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Results

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Naive Bayes (BoW) | 1.0000 | 1.0000 |
| Logistic Regression (TF-IDF) | 1.0000 | 1.0000 |

> **Note:** The dataset is synthetic and contains repeated messages, leading to near-perfect scores. Real-world performance on unseen text would be lower.

## Key Concepts Covered

- **Bag of Words** — frequency-based document vectors; simple but ignores word order.
- **TF-IDF** — downweights common words; better for distinguishing documents.
- **Integer Sequences** — preserve order; required input for RNN/LSTM.
- **LSTM** — gated architecture that mitigates vanishing gradients in long sequences.
- **Attention / Transformers** — parallel, long-range sequence modelling; foundation of modern LLMs.
