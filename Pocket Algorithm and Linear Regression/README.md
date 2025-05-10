# ECE421 Assignment 1: Pocket Algorithm and Linear Regression

This repository contains the implementation of fundamental machine learning algorithms as part of the University of Toronto's **ECE421: Introduction to Machine Learning** course.

## 🧠 Algorithms Implemented
### 1. Pocket Algorithm (Binary Classification on IRIS Dataset)
- **Custom Implementation** using NumPy
- **Benchmark Comparison** using `scikit-learn`'s `Perceptron`
- Evaluated with confusion matrix

### 2. Linear Regression (Regression on Diabetes Dataset)
- **Custom Implementation** using least squares with NumPy
- **Robustness Handling** via pseudo-inverse for singular matrices
- **Benchmark Comparison** using `scikit-learn`'s `LinearRegression`
- Evaluated with mean squared error (MSE)

## 📁 Files
- `PerceptronImp.py`: Pocket Algorithm implementation and evaluation
- `LinearRegressionImp.py`: Linear Regression implementation and evaluation

## ✅ Features
- Adheres to provided function prototypes for automated testing
- Compares custom implementations against `scikit-learn` models
- Includes confusion matrix and MSE for performance evaluation
- Robust handling of singular matrices with `np.linalg.pinv`

## 🔧 How to Run
You can execute each part by running the test functions at the bottom of each file:

```bash
python PerceptronImp.py
python LinearRegressionImp.py
```

## 💡 Environment
- Python 3.8+
- NumPy
- scikit-learn
- Recommended: Google Colab for hassle-free execution

## 📝 Author
Implemented as part of coursework for ECE421 at the University of Toronto.
