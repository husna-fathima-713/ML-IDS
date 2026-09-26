# ML-Based Intrusion Detection System

A Machine Learning based Intrusion Detection System (ML-IDS) for classifying network traffic as **Normal** or **Attack** using the NSL-KDD dataset.

The project compares multiple machine learning models, evaluates them using security-focused metrics, and deploys the selected Random Forest model through a FastAPI API with a real-time monitoring dashboard.

---

## Project Objectives

- Detect malicious network traffic using machine learning.
- Classify traffic into Normal and Attack categories.
- Compare Decision Tree, Random Forest, and SVM models.
- Evaluate performance using precision, recall, F1-score, FPR, FNR, and PR-AUC.
- Analyse classification thresholds and false-positive/false-negative trade-offs.
- Test model behaviour on unseen attack categories.
- Deploy the trained model through a FastAPI prediction API.
- Provide a browser-based monitoring dashboard.

---

## Dataset

The project uses the **NSL-KDD** intrusion detection dataset.

The original attack labels are converted into a binary target:

- `0` → Normal
- `1` → Attack

After preprocessing and duplicate removal:

- Total records: **125,964**
- Normal: **67,343**
- Attack: **58,621**
- Training samples: **100,771**
- Testing samples: **25,193**
- Processed features: **121**

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Removing duplicate records.
2. Converting attack labels into binary classes.
3. Separating categorical and numerical features.
4. One-hot encoding categorical features.
5. Standardizing numerical features.
6. Performing a stratified 80/20 train-test split.
7. Fitting preprocessing transformations only on the training data.

Categorical features include:

- Protocol type
- Service
- Flag

The model uses network-flow characteristics such as duration, source/destination bytes, packet counts, failed login attempts, error rates, and host behaviour.

---

## Machine Learning Models

Three models were evaluated:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Class-weighted learning was used to reduce the effect of class imbalance without creating synthetic network traffic.

### Model Results

| Model | Precision | Recall | F1-Score | FPR | FNR | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 99.75% | 99.90% | 99.83% | 0.22% | 0.10% | 0.997 |
| Random Forest | 99.95% | 99.90% | 99.92% | 0.04% | 0.10% | 1.000 |
| SVM | 99.03% | 99.37% | 99.20% | 0.85% | 0.63% | 0.999 |

The Random Forest model was selected for the deployment prototype based on its measured performance in this experiment.

---

## Threshold Analysis

The deployed model uses a probability threshold of **0.5**.

At this threshold:

- True Positives: 11,713
- True Negatives: 13,463
- False Positives: 6
- False Negatives: 11
- Precision: 99.95%
- Recall: 99.91%
- FPR: 0.0445%
- FNR: 0.0938%

The threshold can be adjusted depending on the required balance between missed attacks and false alerts.

---

## Unseen Attack Evaluation

The model was additionally evaluated on attack categories that were not present in the training data.

The evaluation contained:

- 17 unseen attack categories
- 3,750 attack samples
- Attack recall: 30%
- Attack F1-score: 0.46

This experiment demonstrates an important limitation of supervised intrusion detection: strong performance on known attack patterns does not guarantee strong detection of previously unseen attack behaviour.

Possible production improvements include continuous data collection, analyst feedback, threat intelligence, and periodic model retraining.

---

## System Architecture

```text
Network Traffic
      ↓
Data Collection
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Random Forest Model
      ↓
Risk Score
      ↓
Decision Threshold
      ↓
ALLOW / BLOCK
      ↓
Security Analyst
      ↓
Feedback
      ↓
Model Retraining
```
---

## Dashboard

The deployment dashboard provides:

Live event monitoring
Risk activity graph
Threat-level indicator
Risk score gauge
Traffic decision pipeline
Attack/Normal classification
ALLOW/BLOCK decision
Attack simulation mode
Deployment Screenshots

## Project Structure
```
ML-IDS/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dashboard/
│   └── index.html
│
├── notebooks/
│
├── report/
│
├── results/
│   ├── model/
│   ├── confusion_matrices.png
│   ├── threshold_tradeoff.png
│   ├── architecture.png
│   ├── model_metrics.csv
│   └── final_model_comparison.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── threshold_analysis.py
│   ├── unseen_attack_test.py
│   ├── final_results.py
│   ├── create_architecture.py
│   ├── save_model.py
│   ├── api.py
│   └── test_api.py
│
├── .gitignore
└── README.md
```
---
## Running the Project

1. Activate the virtual environment
```
source .venv/bin/activate
```
3. Start the FastAPI backend
```
python -m uvicorn src.api:app --reload
```
The API runs at:
```
http://127.0.0.1:8000
```
3. Start the dashboard

In another terminal:
```
python -m http.server 5500 --directory dashboard
```
Open:
```
http://127.0.0.1:5500
```
---
## Technologies Used

Python
Pandas
NumPy
Scikit-learn
FastAPI
Uvicorn
Joblib
HTML
CSS
JavaScript
NSL-KDD Dataset

---
## Limitations

The current implementation is a deployment prototype. Network traffic is simulated through the dashboard rather than captured directly from a physical network interface.

The BLOCK decision is an ML classification result and is not connected to a real firewall.

Future development could integrate:

Live packet/flow capture
Network interface monitoring
Firewall enforcement
Persistent alert storage
Authentication
Production monitoring
Continuous model retraining
Additional unseen-attack datasets
Conclusion

This project demonstrates an end-to-end machine learning intrusion detection pipeline, from network traffic preprocessing and model evaluation to API-based deployment and dashboard visualization.

The experiments also show that high performance on known attack categories does not guarantee generalization to unseen attacks, highlighting the importance of continuous evaluation and model improvement in practical intrusion detection systems.
