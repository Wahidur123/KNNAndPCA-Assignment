# KNN & PCA Assignment

## Overview
This assignment covers the implementation of **K-Nearest Neighbors (KNN)** and **Principal Component Analysis (PCA)** in machine learning. The goal is to understand how these techniques work, evaluate their performance, and analyze their impact on datasets.

## Objectives
- Implement and train a **KNN Classifier and KNN Regressor**.
- Compare different distance metrics (Euclidean, Manhattan) in KNN.
- Tune hyperparameters such as **K-value, weights, and leaf_size** in KNN.
- Apply **feature scaling** and observe its effect on KNN performance.
- Implement **KNN Imputation** for handling missing values.
- Train a **PCA model** and analyze explained variance.
- Apply PCA before KNN classification and compare accuracy.
- Visualize PCA-transformed data and decision boundaries of KNN.

## Implemented Tasks
### KNN Implementation
- Train a **KNN Classifier** on synthetic and real-world datasets (Iris, Wine).
- Evaluate using **Precision, Recall, F1-score, and ROC-AUC**.
- Train a **KNN Regressor** and analyze prediction error using **Mean Squared Error (MSE)**.
- Compare performance using **KD Tree and Ball Tree algorithms**.
- Visualize KNN **decision boundaries**.
- Perform **Hyperparameter tuning** using `GridSearchCV`.

### PCA Implementation
- Train a PCA model and analyze **explained variance ratio**.
- Visualize **Scree plot** for high-dimensional datasets.
- Apply PCA before KNN classification and compare accuracy.
- Visualize data projection onto **principal components**.
- Analyze the effect of different **component numbers** on accuracy.
- Compute **reconstruction error** after reducing dimensions.

## Prerequisites
- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `seaborn`, `sklearn`
- Install dependencies using:
  ```bash
  pip install numpy matplotlib seaborn scikit-learn
  ```

## Usage
1. Run the Python scripts for KNN and PCA experiments.
2. Modify parameters like `n_neighbors` (K in KNN) or `n_components` (PCA) to observe changes.
3. Use visualization functions to analyze decision boundaries and variance.

## Results & Observations
- **Feature scaling significantly affects KNN performance.**
- **Higher K values result in smoother decision boundaries but may reduce sensitivity.**
- **PCA helps in dimensionality reduction but can lead to information loss.**
- **Applying PCA before KNN may improve efficiency but not always accuracy.**

## Conclusion
This assignment demonstrates the practical applications of KNN and PCA, highlighting their strengths and limitations. Understanding these techniques is crucial for improving machine learning models in classification and dimensionality reduction tasks.

