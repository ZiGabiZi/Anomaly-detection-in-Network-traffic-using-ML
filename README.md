# Anomaly Detection in Network Traffic Using Machine Learning

This project explores anomaly detection in network traffic using machine learning techniques. The goal is to develop a system capable of identifying unusual or potentially malicious activities in network traffic, leveraging the UNSW-NB15 dataset and machine learning models like Autoencoders, Random Forest, Decision Trees, and Extra Trees Classifiers.

---

## Features
- Implements anomaly detection using an Autoencoder in PyTorch.
- Uses ensemble methods (`RandomForestClassifier`, `DecisionTreeClassifier`, `ExtraTreesClassifier`) with a voting classifier for evaluation.
- Includes feature selection with Recursive Feature Elimination (RFE).
- Dataset preprocessing with PCA for dimensionality reduction.
- Visualization of reconstruction loss distributions using `seaborn`.
- Brute-force optimization for threshold selection to maximize detection accuracy.
- Pre-trained model support with `.pth` files for inference and further training.

---

## Dataset
The project utilizes the [UNSW-NB15 dataset](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html), a publicly available collection of network traffic records, which includes normal and attack data. Features from the dataset are used to train and evaluate the models.

- **Dataset Features:**
  - Contains 49 attributes, including flow features, basic features, and content-based features.
  - Attack categories: `DoS`, `Fuzzers`, `Worms`, `Backdoors`, etc.
  - Balanced split between normal and malicious traffic for training/testing.

---

## Dependencies
The project requires the following Python libraries:
- `torch`
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

Install them using:
```bash
pip install -r requirements.txt
