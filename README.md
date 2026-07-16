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

## 📈 Performance Summary & Comparison

Below is the comparative analysis of both approaches. It highlights how supervised models excel when labeled historical data is available, while unsupervised models serve as a strong baseline when labels are scarce or when detecting completely new fraud patterns.

### Model Performance

| Approach | Model | Test Precision | Test Recall | Best F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **Supervised** | Logistic Regression (Baseline) | 5.00% | 87.76% | 9.46% |
| **Supervised** | Random Forest (Baseline) | 26.62% | 83.67% | 40.39% |
| **Supervised** | **XGBoost (Optimized via CUDA)** | **95.24%** | **81.63%** | **87.91%** |
| **Unsupervised** | Isolation Forest (Anomaly Detection) | 17.91% | 73.98% | 28.84% |

> *Note: While the Supervised XGBoost achieves the highest overall F1-Score (87.91%), the Unsupervised Isolation Forest serves as an essential defense line to catch "zero-day" fraud attacks that **do not look like past fraud patterns**.*

> *Note: In fraud detection, we intentionally optimize for a higher **Recall** to catch more fraudulent cases, accepting a minor, calculated trade-off in Precision.*

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
