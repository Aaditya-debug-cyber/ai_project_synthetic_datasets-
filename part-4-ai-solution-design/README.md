# Part 4 – AI Solution Design for a Business Problem

## Overview

This repository contains the complete AI solution design for an **automated chest X-ray triage system** in the **Healthcare** domain. The solution uses a pre-trained CNN (DenseNet-121) with transfer learning to classify incoming X-rays as Normal, Abnormal-Priority, or Abnormal-Urgent, enabling intelligent radiologist worklist prioritisation.

## Selected Domain & Problem

| Field | Value |
|---|---|
| Domain | Healthcare |
| Problem | Manual X-ray triage creates delays for urgent findings; 6.9% error rate and 30+ hour resolution times at baseline |
| AI Task | Multi-class image classification |
| Model | DenseNet-121 (transfer learning from ImageNet) |

## Repository Structure

```
part-4-ai-solution-design/
│
├── README.md                         ← this file
├── solution_report.md                ← full 8-task solution design document
└── diagrams/
    └── solution_architecture.png     ← end-to-end architecture diagram
```

## Tasks Covered

| Task | Topic |
|------|-------|
| 1 | Business domain selection (Healthcare) |
| 2 | Problem definition — stakeholders, current process, limitations |
| 3 | AI task type — image classification with justification |
| 4 | Data requirement plan — types, features, labels, quality risks |
| 5 | Model recommendation — DenseNet-121 with transfer learning |
| 6 | Evaluation plan — technical metrics, business KPIs, failure cases, human review |
| 7 | Responsible AI — bias, privacy, over-reliance, workforce impact, human oversight |
| 8 | One-page solution summary |

## Expected Business Impact

Based on the `business_kpi_sample.csv` baseline data:

| KPI | Baseline | Target |
|---|---|---|
| Manual processing hours / month | 454 hrs | ≤ 180 hrs (↓ 60 %) |
| Average resolution time | 30 hrs | ≤ 12 hrs (↓ 60 %) |
| Error rate | 6.9 % | ≤ 2.5 % (↓ 64 %) |
| Satisfaction score | 7.1 / 10 | ≥ 8.5 / 10 |

## Key Design Decisions

- **Transfer learning** chosen because medical imaging datasets are small relative to general image datasets; ImageNet pre-training provides robust low-level features.
- **DenseNet-121** chosen specifically because of its proven performance on chest X-ray classification (CheXNet benchmark).
- **Human-in-the-loop** architecture ensures AI is a Clinical Decision Support tool only — radiologist sign-off is always mandatory.
- **Confidence gating** routes low-confidence predictions to mandatory human review.

## Reference Data Used

- `ai_usecase_reference_catalog.csv` — Healthcare row informed domain, task type, model choice, and risk identification.
- `business_kpi_sample.csv` — Provided the quantitative baseline for all business KPI targets.
