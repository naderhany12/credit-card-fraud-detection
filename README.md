# Credit Card Fraud Detection: Supervised & Anomaly Detection Pipelines

An end-to-end, high-performance Machine Learning project tackling the classic challenge of highly imbalanced credit card transaction data. This repository features two distinct approaches: **Supervised Classifiers** (optimized via GPU/CUDA) and **Unsupervised Anomaly Detection** techniques.

---

## 🚀 Key Highlights & Engineering Choices
* **CUDA-Accelerated GridSearch:** Integrated `XGBoost` with `tree_method='hist'` and `device='cuda'` to leverage GPU acceleration, reducing hyperparameter tuning time from hours to minutes.
* **Smart Preprocessing:** Handled severe data skewness in continuous features (such as `Amount` and `V1-V28` PCA components) using Scikit-Learn's `PowerTransformer` (Yeo-Johnson).
* **Zero Data Leakage:** Pipeline strictly adheres to fitting the transformers/scalers only on the training set, while applying transforms on the validation and test sets.
* **Class Imbalance Mitigation:** Leveraged `stratify` splits and tuned the `scale_pos_weight` parameter in XGBoost to handle the extreme class imbalance (~0.17% fraud rate) effectively.

---

## 📊 Project Architecture & Approaches

### 1. Supervised Learning Pipeline
A comparative analysis between baseline and optimized classifiers:
* **Random Forest (Baseline):** Established a reliable baseline.
* **XGBoost (CUDA Optimized):** Tuned systematically via `GridSearchCV` on GPU to maximize the overall F1-Score while prioritizing **Recall** (catching the maximum number of fraudulent transactions).

### 2. Unsupervised Anomaly Detection Pipeline (Incoming/Included)
Tackles the fraud detection challenge by treating fraudulent transactions as rare anomalies/outliers, using:
* **Isolation Forest** or **Local Outlier Factor (LOF)** to model the boundary of "normal" behavior and flag anomalies without relying heavily on labeled fraud data.

---

## 📈 Performance Summary (Supervised Models)

| Model | Test Precision | Test Recall | Best F1-Score |
| **Logistic Regression(Baseline)** | 5.0%  | 87.76% | 9.46% |
| **Random Forest (Baseline)** | 26.62% | 83.67 % | 40.39% |
| **XGBoost (Optimized via CUDA)** | **95.24%** | **81.63%** | **87.91%** |

> *Note: In fraud detection, we intentionally optimize for a higher **Recall** (83.6%) to catch more fraudulent cases, accepting a minor, calculated trade-off in Precision.*

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.12
* **Hardware Acceleration:** NVIDIA GPU (CUDA v12+)
* **Libraries:** `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📁 Repository Structure
```text
├── notebooks/
│   ├── fraud-detection-supervised-models.ipynb   # EDA, PowerTransformer, and CUDA XGBoost
│   └── fraud-detection-anomaly-detection.ipynb   # Isolation Forest / Unsupervised models
└── README.md                                     # Project Documentation
