# Part 4 – AI Solution Design Report
## Automated Chest X-Ray Triage using Convolutional Neural Networks

**Domain:** Healthcare  
**Prepared by:** AI Business Analyst  
**Reference files used:** `ai_usecase_reference_catalog.csv`, `business_kpi_sample.csv`

---

## Task 1 – Business Domain

**Selected Domain: Healthcare**

Healthcare is selected because it combines high data availability (medical imaging archives), clear human cost of errors, and a well-established AI research track record. The catalog entry for this domain identifies *medical image triage* as the target problem, with image classification as the task type and CNN / transfer learning as the recommended architecture. The KPI sample data provides a concrete baseline of operational performance against which AI impact can be measured.

---

## Task 2 – Business Problem Definition

### What problem is being solved?

Hospitals and diagnostic centres receive hundreds of chest X-rays every day. Radiologists must manually review every image to determine whether it shows signs of abnormality (e.g. pneumonia, pleural effusion, cardiomegaly, pneumothorax). In high-volume settings this creates significant delays — urgent cases may sit in the queue for hours before a clinician reaches them.

The proposed solution is an **AI-powered triage assistant** that automatically classifies each incoming chest X-ray as:
- **Normal** — no abnormality detected; can be deprioritised.
- **Abnormal – Priority** — findings present; should be reviewed within 2–4 hours.
- **Abnormal – Urgent** — life-threatening finding present (e.g. large pneumothorax); radiologist alerted immediately.

### Who are the users and stakeholders?

| Stakeholder | Role |
|---|---|
| Radiologists | Primary users — receive sorted worklists and AI annotations |
| Emergency physicians | Benefit from faster turnaround on urgent scans |
| Hospital administrators | Interested in cost reduction and throughput |
| Patients | Benefit from faster diagnosis and treatment |
| Compliance / Legal | Concerned with liability, audit trail, and data privacy |

### Current manual process

1. X-ray taken by radiographer; DICOM image sent to PACS system.
2. Image joins radiologist worklist in order of arrival.
3. Radiologist manually reviews each image, dictates report.
4. Report transcribed, reviewed, and sent to referring physician.
5. Average turnaround: **30+ hours** (from KPI baseline data).

### Limitations of the current process

- **No intelligent prioritisation** — a life-threatening finding may sit behind routine follow-ups.
- **High manual processing load** — baseline data shows ~454 manual processing hours per month.
- **Error rate** — baseline average error rate of 6.9 %, attributable to fatigue and high volume.
- **Low satisfaction scores** — baseline patient/clinician satisfaction of 7.1 / 10.
- **Not scalable** — adding imaging capacity requires proportional radiologist headcount.

---

## Task 3 – AI Task Type

**Task type: Image Classification**

Each chest X-ray is a single 2D image that must be assigned to one of three discrete categories (Normal / Abnormal Priority / Abnormal Urgent). This is a supervised multi-class image classification problem.

**Why this is appropriate:**

| Reason | Explanation |
|---|---|
| Fixed output set | There are a finite, well-defined set of triage outcomes. |
| Visual input | The signal is entirely encoded in pixel patterns — spatial feature extraction is essential. |
| Supervised labels available | Radiologist-annotated datasets (e.g. CheXNet, NIH Chest X-ray14) provide ground-truth labels. |
| Probability output | Softmax classification gives a confidence score useful for threshold-based human review triggers. |

Alternative task types considered and rejected:
- **Object detection** — would locate specific lesions within the image; useful as a downstream step but not required for triage.
- **Anomaly detection** — suitable if labels were unavailable; here we have labelled data.
- **Segmentation** — useful for precise lesion measurement; beyond the scope of a triage tool.

---

## Task 4 – Data Requirement Plan

### Type of data needed

| Data type | Description | Format |
|---|---|---|
| Chest X-ray images | Frontal (PA/AP) chest radiographs | DICOM, converted to PNG/JPEG |
| Patient metadata | Age, sex, clinical indication | Structured CSV / EHR export |
| Radiologist annotations | Triage class label per image | CSV label file |
| Historical reports | Free-text radiology reports for NLP-based label extraction | Text |

### Structured vs. unstructured

- **Unstructured (primary input):** X-ray images — high-dimensional pixel arrays requiring CNN processing.
- **Structured (auxiliary):** Patient metadata (age, sex, prior history) that can be fused at the classification head to improve performance.

### Input features

- Raw pixel values of the 224×224 normalised grayscale X-ray image.
- Patient age (binned into groups: paediatric, adult, elderly).
- Biological sex (one-hot encoded).
- Clinical indication text (optionally embedded via a pre-trained text encoder).

### Target variable / labels

`triage_class` — a three-way categorical label:
- `0` = Normal
- `1` = Abnormal – Priority
- `2` = Abnormal – Urgent

### Data collection method

1. **Retrospective data:** Extract de-identified DICOM images from hospital PACS for the past 3–5 years. Match to radiologist reports using accession numbers.
2. **Label generation:** Use NLP to extract triage-relevant phrases from reports ("no acute cardiopulmonary disease" → Normal; "pneumothorax" → Urgent). Reviewed by a senior radiologist.
3. **Public datasets:** Augment with NIH ChestX-ray14 (112,000 images), CheXPert (224,000 images), and MIMIC-CXR.
4. **Prospective collection:** After deployment, capture radiologist corrections as ongoing labelled data for continuous retraining.

### Data quality risks

| Risk | Mitigation |
|---|---|
| Label noise from NLP extraction | Senior radiologist review of 10 % sample; inter-annotator agreement measured |
| Class imbalance (Urgent cases rare) | SMOTE oversampling + class-weighted loss function |
| Demographic bias (e.g. dataset skewed toward certain age groups) | Audit label distribution across age, sex, and ethnicity |
| DICOM metadata inconsistency | Standardise using `pydicom` with strict field validation |
| Image quality variation (rotation, exposure) | Data augmentation + CLAHE pre-processing |

---

## Task 5 – Model Recommendation

### Recommended architecture: Transfer Learning with DenseNet-121

**Why DenseNet-121?**

DenseNet-121 was the backbone of CheXNet (Rajpurkar et al., 2017), which demonstrated radiologist-level performance on chest X-ray pathology detection. Its dense connectivity pattern (each layer receives feature maps from all preceding layers) is particularly effective for medical images because:

- It maximises gradient flow, which is critical for fine-tuning on relatively small medical datasets.
- It reuses features at multiple scales, capturing both fine-grained textures and large structural patterns.
- It has fewer parameters than equivalent ResNets, reducing overfitting risk.

### Architecture design

```
Input Image  (224 × 224 × 1 grayscale / 3-channel repeat)
       ↓
DenseNet-121 Backbone
  • Pre-trained on ImageNet
  • Freeze all layers initially for warm-up (5 epochs)
  • Unfreeze Dense Block 4 + Transition Layer 3 for fine-tuning
       ↓
Global Average Pooling  (1024-dim feature vector)
       ↓
Dropout (p = 0.4)
       ↓
Dense (256 units, ReLU)
       ↓
Dropout (p = 0.3)
       ↓
Dense (3 units, Softmax)  →  P(Normal), P(Priority), P(Urgent)
```

### Training configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam (warm-up LR 1e-4 → 1e-5 during fine-tuning) |
| Loss function | Categorical cross-entropy with class weights |
| Batch size | 32 |
| Epochs | 30 (with early stopping, patience = 5) |
| Data augmentation | Random horizontal flip, rotation ±10°, brightness ±0.1, zoom ±10 % |
| Input size | 224 × 224 |

### Why transfer learning specifically?

Medical imaging datasets are expensive to label and smaller than general image datasets. Transfer learning from ImageNet gives the model robust low-level feature detectors (edges, textures) for free, dramatically reducing the amount of labelled X-ray data needed and improving generalisation.

---

## Task 6 – Evaluation Plan

### Technical metrics

| Metric | Why it matters |
|---|---|
| **AUC-ROC** | Measures discriminative power across all classification thresholds |
| **Sensitivity (Recall) for Urgent class** | Missing an urgent case is clinically catastrophic — must be maximised |
| **Specificity** | Excessive false alarms exhaust radiologist attention |
| **Weighted F1-score** | Balances precision and recall across imbalanced classes |
| **Top-1 Accuracy** | Overall classification accuracy across all three classes |

**Target thresholds:**
- Sensitivity for Urgent class ≥ 0.97 (clinical requirement: miss fewer than 3 in 100 urgent cases).
- Specificity for Normal class ≥ 0.90 (avoid unnecessary radiologist workload).

### Business metrics

These are derived from the KPI baseline data (`business_kpi_sample.csv`):

| KPI | Baseline (avg) | Target with AI |
|---|---|---|
| Manual processing hours / month | 454 hours | ≤ 180 hours (↓ 60 %) |
| Average resolution time | 30.0 hours | ≤ 12 hours (↓ 60 %) |
| Error rate | 6.9 % | ≤ 2.5 % (↓ 64 %) |
| Satisfaction score | 7.1 / 10 | ≥ 8.5 / 10 |

### Possible failure cases

| Failure | Consequence | Guard |
|---|---|---|
| High-confidence miss of urgent pneumothorax | Patient harm | Confidence threshold forces review below 0.85 |
| Incorrect Normal classification for a paediatric image | Delayed treatment | Paediatric sub-model or demographic stratified testing |
| Model degradation over time (distribution shift) | Silent performance loss | Monthly AUC monitoring + drift alerts |
| Adversarial inputs / corrupted DICOMs | Crash or misclassification | Input validation pipeline with format checks |

### Human review and validation process

1. **Confidence gate:** Any prediction below 0.85 confidence is automatically queued for radiologist review regardless of predicted class.
2. **Urgent override:** All Urgent predictions trigger an alert to the on-call radiologist — AI cannot autonomously act; it only prioritises.
3. **Prospective validation:** First three months of deployment are shadow mode — AI predictions logged but worklist is not re-sorted. AUC compared against retrospective ground truth before full deployment.
4. **Quarterly audit:** Random 5 % sample of Normal predictions reviewed by a senior radiologist to detect silent misses.

---

## Task 7 – Responsible AI Considerations

### Bias in data

Chest X-ray datasets historically over-represent certain demographics (e.g. adult males, Western populations). A model trained on such data may under-perform for women, elderly patients, or patients from under-represented ethnic groups where pathology presents differently.

**Mitigation:** Conduct pre-training fairness audit. Measure AUC stratified by age group, sex, and ethnicity. If disparity exceeds 3 %, collect additional data from under-represented groups before deployment.

### Incorrect predictions

False negatives (missing urgent findings) carry direct patient harm. False positives (over-alerting radiologists) cause alert fatigue and erode trust.

**Mitigation:** Clinical validation study with ≥ 500 cases reviewed by three independent radiologists. Publish sensitivity/specificity at the chosen operating threshold. Set the operating threshold conservatively to prioritise recall for the Urgent class.

### Privacy concerns

X-ray images, even de-identified, can potentially be re-identified when combined with age, sex, and clinical indication.

**Mitigation:** Strict HIPAA / GDPR-compliant de-identification (remove all 18 HIPAA identifiers from DICOM metadata). Data access controlled by role-based access control. Model outputs stored in the hospital's internal network — no image data leaves the hospital perimeter.

### Over-reliance on AI

Clinicians who see high AI accuracy may begin to rubber-stamp AI predictions without critical assessment, especially under workload pressure.

**Mitigation:** Training programme for radiologists covering the model's known failure modes. UI design deliberately requires the radiologist to document their independent interpretation before the AI suggestion is displayed (two-step review). Regular workshops where AI errors are discussed.

### Impact on users (workforce)

Junior radiologists may lose diagnostic skill development if they defer to AI, and there are concerns about job displacement.

**Mitigation:** Position AI as a workload tool, not a replacement. Ensure AI handles only triage prioritisation; all diagnostic reports remain the responsibility of the radiologist. Document and communicate the policy clearly to radiology staff.

### Human oversight

AI must never make a final clinical decision autonomously.

**Mitigation:** The system is architected as a **Clinical Decision Support (CDS) tool**, not an autonomous diagnostic system. Every patient record includes a mandatory radiologist sign-off field that cannot be pre-populated by AI. All predictions are logged with model version, timestamp, and confidence score for regulatory audit.

---

## Task 8 – Final Solution Summary

### One-Page Summary

---

| Section | Detail |
|---|---|
| **Domain** | Healthcare |
| **Problem** | Manual chest X-ray triage creates dangerous delays for urgent cases, high workload, and a 6.9 % error rate at baseline |
| **Stakeholders** | Radiologists, emergency physicians, hospital administrators, patients |
| **AI Task** | Multi-class image classification (Normal / Priority / Urgent) |
| **Proposed Solution** | CNN-based triage assistant using DenseNet-121 with transfer learning, deployed as a CDS tool within the hospital PACS workflow |
| **Required Data** | Labelled chest X-ray DICOM images (retrospective + NIH/CheXPert public datasets), patient metadata, radiologist annotations |
| **Model** | DenseNet-121 (ImageNet pre-trained) → fine-tuned on chest X-ray data → 3-class Softmax output |
| **Training approach** | Transfer learning with progressive unfreezing, class-weighted loss, data augmentation |
| **Technical KPIs** | AUC-ROC ≥ 0.93, Urgent recall ≥ 0.97, Normal specificity ≥ 0.90 |
| **Business KPIs** | Manual hours ↓ 60 %, resolution time ↓ 60 %, error rate ↓ 64 %, satisfaction ↑ to 8.5/10 |
| **Deployment** | Shadow mode (3 months) → clinical validation → phased rollout → full deployment |
| **Key risks** | Demographic bias, missed urgent findings, over-reliance, privacy |
| **Mitigations** | Fairness audits, confidence gating, human-in-the-loop architecture, HIPAA/GDPR compliance, quarterly review |
| **Regulatory** | FDA 510(k) clearance pathway (Software as a Medical Device); CE marking if EU deployment |

---

### Architecture Diagram

See `diagrams/solution_architecture.png` for the full end-to-end architecture covering data sources, preprocessing, model layers, evaluation, clinical action, and responsible AI safeguards.

---

*This solution design is grounded in the AI Use Case Reference Catalog (Healthcare row: image classification, CNN/transfer learning, AUC + sensitivity metrics, privacy and human review risks) and validated against the business KPI sample showing baseline manual processing hours, resolution times, error rates, and satisfaction scores.*
