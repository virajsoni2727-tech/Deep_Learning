# 🧠 Deep Learning PR2 — Adult Income Classification

## Comparative Analysis of Deep Learning Techniques

This project uses an Artificial Neural Network (ANN) to classify whether an individual's annual income falls into:

- **<=50K**
- **>50K**

using the **Adult Income (Census Income) dataset**.

The project follows the PR2 requirements and evaluates how different deep-learning choices influence model performance.

---

## 🔄 Project Scope

The original project concept and experimental workflow are retained. The changes in this version focus on presentation, wording, organization, and readability rather than replacing the underlying methodology.

## 📌 Project Goal

The primary goal is to develop a binary-classification neural network and systematically evaluate:

1. Activation Functions
2. Weight Initialization Techniques
3. Loss Functions
4. Batch Normalization
5. Optimizers
6. Learning Rates

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Since the target classes are imbalanced, model quality is assessed using more than accuracy alone.

---

## 📊 Dataset

**Dataset:** Adult Income / Census Income Dataset

**Task:** Binary Classification

**Target:** `income`

| Target | Meaning |
|---|---|
| `<=50K` | Income is at or below $50K |
| `>50K` | Income is above $50K |

### Main Features

The dataset contains demographic and employment-related features such as:

- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

The project removes redundant columns such as `fnlwgt` and `education.num` during preprocessing, following the project specification.

---

## 🏗️ Project Pipeline

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Remove Redundant Columns
   ↓
Target Encoding
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Baseline ANN
   ↓
Activation Function Experiment
   ↓
Weight Initialization Experiment
   ↓
Loss Function Experiment
   ↓
Batch Normalization Experiment
   ↓
Optimizer Experiment
   ↓
Learning Rate Sensitivity
   ↓
Final Model Comparison
   ↓
Best Model
```

---

# 📁 Repository Structure

```text
DL_PR2_Adult_Income/
│
├── DL_PR2.ipynb
├── adult.csv
├── requirements.txt
├── README.md
├── final_results.csv
├── final_model_ranking.csv
│
├── plots/
│   ├── class_distribution.png
│   ├── baseline_accuracy.png
│   ├── baseline_loss.png
│   ├── activation_comparison.png
│   ├── initialization_comparison.png
│   ├── loss_comparison.png
│   ├── batchnorm_comparison.png
│   ├── optimizer_comparison.png
│   ├── learning_rate.png
│   └── roc_curve.png
│
└── video/
    └── demo_link.txt
```

> **Important:** The image files below should be placed inside the `plots/` folder after you generate them from the notebook. If a plot has not been generated yet, the README image will not display until the file is added.

---

# 🔬 Task 1 — Data Loading, Cleaning & EDA

The first stage loads, cleans, and prepares the Adult Income dataset for modeling.

### Preprocessing performed

- Load the dataset
- Inspect shape and data types
- Remove leading/trailing whitespace
- Convert `?` values to missing values
- Handle missing categorical values
- Remove redundant columns
- Encode the target variable
- Analyze class distribution
- One-hot encode categorical features
- Standardize numerical features
- Split data into training and testing sets

### 📈 Class Distribution

![Class Distribution](plots/class_distribution.png)

---

# 🤖 Task 2 — Baseline ANN

A baseline Artificial Neural Network is established first and then used as the reference point for the experiments.

### Baseline Architecture

```text
Input Features
      ↓
Dense(128)
      ↓
ReLU
      ↓
Dense(64)
      ↓
ReLU
      ↓
Dense(1)
      ↓
Sigmoid
      ↓
Income Prediction
```

### Baseline Configuration

| Parameter | Value |
|---|---|
| Hidden Layers | 128, 64 |
| Activation | ReLU |
| Output Activation | Sigmoid |
| Loss | Binary Cross Entropy |
| Optimizer | Adam |
| Batch Size | 256 |
| Epochs | 50 |

### 📈 Baseline Accuracy

![Baseline Accuracy](plots/baseline_accuracy.png)

### 📉 Baseline Loss

![Baseline Loss](plots/baseline_loss.png)

---

# ⚡ Task 3 — Activation Function Comparison

Four activation functions are compared:

- ReLU
- Tanh
- Sigmoid
- ELU

### 📊 Activation Comparison

![Activation Comparison](plots/activation_comparison.png)

This experiment examines validation performance together with training/convergence behavior.

---

# 🎲 Task 4 — Weight Initialization

The following initialization methods are evaluated:

- Zeros
- Random Normal
- He Normal
- Glorot Uniform

### 📊 Initialization Comparison

![Initialization Comparison](plots/initialization_comparison.png)

This experiment examines how different starting weight distributions influence learning and convergence.

---

# 📉 Task 5 — Loss Function Comparison

The project compares:

1. Binary Cross Entropy
2. Mean Squared Error
3. Weighted Binary Cross Entropy
4. Focal Loss

### 📊 Loss Comparison

![Loss Comparison](plots/loss_comparison.png)

Weighted BCE and Focal Loss are used to examine how loss design can address class imbalance.

---

# 🧪 Task 6 — Batch Normalization

Two configurations are compared:

- ANN without Batch Normalization
- ANN with Batch Normalization

### 📊 Batch Normalization Comparison

![Batch Normalization Comparison](plots/batchnorm_comparison.png)

The comparison considers validation performance as well as training convergence.

---

# 🚀 Task 7 — Optimizer Comparison

The following optimizers are compared:

- SGD
- SGD + Momentum
- RMSprop
- Adam

### 📊 Optimizer Comparison

![Optimizer Comparison](plots/optimizer_comparison.png)

---

# 🎯 Learning Rate Sensitivity

Different learning rates are tested to understand how learning-rate selection affects model performance.

Tested values:

```text
0.0001
0.001
0.01
0.1
```

### 📈 Learning Rate Experiment

![Learning Rate Sensitivity](plots/learning_rate.png)

---

# 📊 Evaluation Metrics

Every model is evaluated using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions |
| Recall | Positive cases successfully detected |
| F1-Score | Balance between Precision and Recall |
| ROC-AUC | Ranking/classification performance across thresholds |

---

# 🏆 Final Model Selection

All experiments are combined into a final comparison table.

The final model is selected based on the experimental results rather than assuming a technique is best in advance.

### Final Results

The notebook generates:

- `final_results.csv`
- `final_model_ranking.csv`

Example format:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline ANN | — | — | — | — | — |
| Activation - ReLU | — | — | — | — | — |
| Activation - Tanh | — | — | — | — | — |
| Activation - Sigmoid | — | — | — | — | — |
| Activation - ELU | — | — | — | — | — |
| Initialization - He Normal | — | — | — | — | — |
| Loss - Weighted BCE | — | — | — | — | — |
| Loss - Focal Loss | — | — | — | — | — |
| With BatchNorm | — | — | — | — | — |
| Optimizer - Adam | — | — | — | — | — |

> Replace the `—` values with the actual results generated by `DL_PR2.ipynb`.

---

# 📈 ROC Curve

The notebook also generates a ROC curve for model evaluation.

![ROC Curve](plots/roc_curve.png)

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DL_PR2_Adult_Income
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Open the notebook

```bash
jupyter notebook DL_PR2.ipynb
```

Or open the notebook using **Google Colab**.

## 4. Add the dataset

Place:

```text
adult.csv
```

in the project directory.

## 5. Run the notebook

Run the cells from top to bottom.

---

# 🎥 Project Demonstration Video

The PR2 submission requires a recorded demonstration.

Add your video link below:

**Video:** `PASTE_YOUR_VIDEO_LINK_HERE`

The video should demonstrate:

- Jupyter Notebook / Colab
- Data preprocessing
- Baseline ANN
- Activation experiments
- Weight initialization experiments
- Loss experiments
- Batch Normalization
- Optimizer comparison
- Final results
- Explanations of why the techniques were used

---

# 📦 Requirements

All Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 👩‍💻 Author

**Name:** Disha Lukhi

**Course:** Deep Learning

**Project:** PR2 — Adult Income Prediction

**Institute:** Red & White Skill Education

---

# 📌 Conclusion & Summary

This project demonstrates how different neural-network design choices can affect a binary classification problem.

The experiments provide a systematic comparison of activation functions, weight initialization, loss functions, Batch Normalization and optimizers.

The final model is chosen from the recorded experimental results using multiple evaluation metrics instead of relying on accuracy alone.
