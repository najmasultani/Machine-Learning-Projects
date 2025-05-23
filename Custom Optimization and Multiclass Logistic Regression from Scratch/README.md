# MyTorch: From-Scratch Gradient Descent & Multiclass Logistic Regression

This project is a personal implementation of core machine learning algorithms without relying on external ML libraries like PyTorch or scikit-learn. It was developed for ECE421 (Introduction to Machine Learning) at the University of Toronto to demonstrate a deep understanding of optimization techniques and model training mechanics.

## 🧠 Key Features

### 🔁 Optimization Algorithms (via `Optimizer` class)
- **Stochastic Gradient Descent (SGD)**
- **Heavy-ball Momentum**
- **Nesterov Momentum**
- **Adam Optimizer**

### 🧪 Multiclass Logistic Regression Model
- Full from-scratch implementation (no ML libraries)
- Mini-batch training with various optimizers
- Evaluation metrics: accuracy, cross-entropy loss
- Supports custom learning rates, momentum, and batch size

### 📊 Bonus: K-Means Clustering (template included)

## 📁 File Overview
- `myTorch.py` – Core implementation of optimization methods and logistic regression
- `util.py` – Utility functions and training helpers
- `test.py` – Includes test cases to verify correctness of each module

## 🧪 How to Run
```bash
python test.py
```

This runs a series of unit tests to evaluate convergence behavior, performance of different optimizers, and classification accuracy.

## 📦 Requirements
- Python 3.8+
- NumPy only (No PyTorch, TensorFlow, or scikit-learn used)

## 📝 Author
Developed as a part of ECE421 – Introduction to Machine Learning at the University of Toronto (Fall 2024)
