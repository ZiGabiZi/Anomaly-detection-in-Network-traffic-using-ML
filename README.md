# Anomaly Detection in Network Traffic Using Machine Learning

> ⚠️ **Work in progress** – the project is being refactored. The results of the old version (`notebooks/00_original_reference.ipynb`) are not methodologically valid and should not be used.

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
The project utilizes the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), a publicly available collection of network traffic records, which includes normal and attack data. Features from the dataset are used to train and evaluate the models.

- **Dataset Features:**
  - Contains 49 attributes, including flow features, basic features, and content-based features.
  - Attack categories: `DoS`, `Fuzzers`, `Worms`, `Backdoors`, etc.

---

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell: .venv\Scripts\Activate.ps1)
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
pip install -e .
```

Without an NVIDIA GPU, the CPU-only PyTorch build is much smaller: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Data
The dataset files are not stored in this repository. Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the [UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in `data/raw/`.
