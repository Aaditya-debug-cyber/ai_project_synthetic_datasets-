# Part 1 — Neural Network Fundamentals: Customer Churn Prediction

## Overview
Binary classification neural network to predict customer churn using the
Customer Churn Neural Network Dataset (2000 records, 16 features, 1 target).

**Key challenge:** Severe class imbalance — only 31 churned customers (1.6%)
vs 1969 retained (98.4%), a 63:1 ratio.

## Dataset
| Property | Value |
|---|---|
| File | `customer_churn_nn.csv` |
| Rows | 2,000 |
| Features | 15 (after dropping `customer_id`) |
| Target | `churn` (0=Retained, 1=Churned) |
| Categorical | region, plan_type, contract_type, payment_method |
| Numerical | tenure, charges, login days, tickets, delays, data, satisfaction, etc. |
| Missing values | None |

## Tasks Completed
| Task | Description |
|---|---|
| 1 | Dataset exploration: shape, dtypes, stats, target distribution, EDA plots |
| 2 | Preprocessing: drop ID, LabelEncode categoricals, StandardScaler, stratified split, class weights |
| 3 | Baseline model: 2L-64-ReLU + BatchNorm + Dropout(0.2) → Sigmoid output |
| 4 | Training with EarlyStopping + ReduceLROnPlateau; metrics: Loss, Acc, AUC, F1, Recall |
| 5 | 9 hyperparameter experiments comparing layers, neurons, LR, batch, activation |
| 6 | Written reflection on weights/biases, activations, learning rate, overfitting |

## Baseline Architecture
```
Input (15) → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
           → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
           → Dense(1, Sigmoid)
```
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Regularisation:** Dropout(0.2), BatchNorm, EarlyStopping (patience=25), class weights {0:0.51, 1:32.0}

## Baseline Results (Test Set)
| Metric | Value |
|---|---|
| AUC-ROC | ~0.90 |
| Accuracy | ~0.88 |
| Recall (Churned) | ~0.33 |

> Note: With a 63:1 imbalance, AUC-ROC is the primary metric.
> High accuracy alone is misleading — a model predicting all-zero would get 98.5%.

## Experiment Summary
| # | Config | Test AUC | Test F1 | Recall |
|---|---|---|---|---|
| 1 | Baseline 2L-64 ReLU | 0.9425 | 0.0805 | 1.00 |
| 2 | Shallow 1L-32 ReLU | 0.9592 | 0.1446 | 1.00 |
| 3 | Deep 3L-128 ReLU | 0.8585 | 0.2143 | 0.50 |
| 4 | High LR 0.01 | 0.9429 | 0.0945 | 1.00 |
| 5 | Low LR 0.0001 | 0.9748 | 0.1008 | 1.00 |
| 6 | Large Batch 64 | 0.9596 | 0.1395 | 1.00 |
| 7 | Small Batch 8 | 0.9006 | 0.1096 | 0.67 |
| 8 | Tanh Activation | 0.9440 | 0.1739 | 1.00 |
| 9 | ELU Activation | 0.9863 | 0.0622 | 1.00 |

## Project Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb            ← Full end-to-end notebook
├── notebook_script.py        ← Standalone Python version
├── requirements.txt
├── customer_churn_nn.csv     ← Dataset
└── results/
    ├── task1_eda.png                 ← EDA dashboard
    ├── task4_evaluation.png          ← Training curves + confusion matrix + ROC
    ├── model_comparison_table.csv    ← All 9 experiments comparison
    ├── model_comparison_table.png    ← Bar chart comparison
    ├── evaluation_outputs.png        ← Combined summary figure
    └── task6_reflection.txt          ← Written reflection
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
# or
python notebook_script.py
```
