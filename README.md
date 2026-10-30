# Anomaly Detection in Network Traffic Using Machine Learning

> ⚠️ **Work in progress** – the project is being refactored. The results of the old version (`notebooks/00_original_reference.ipynb`) are not methodologically valid and should not be used.

Detecting malicious network flows on the UNSW-NB15 dataset, with supervised classifiers (attack vs normal, and attack category) and anomaly detection models trained on normal traffic only.

---

## Status
- ✅ Data loading and EDA
- ✅ Preprocessing
- ✅ Evaluation framework
- ✅ Supervised models (logistic regression, random forest, gradient boosting)
- ✅ Anomaly detection: Isolation Forest, autoencoder (PyTorch)
- ⏳ Anomaly detection: leave-one-attack-out
- ⏳ Inference script

---

## Dataset
The project uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (UNSW Canberra): labelled network flows with normal traffic and 9 attack categories (`Fuzzers`, `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`).
The official train/test files contain 42 features per flow, plus `label` (0 normal, 1 attack) and `attack_cat`.

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

### Data handling
- The official train/test split is kept (175,341 / 82,332 flows).
- Train is cleaned separately for each task (`label` for binary, `attack_cat` for multi-class): feature vectors with more than one label keep the majority label (ties are dropped), then only one row per feature vector is kept. 20% of it is then held out as a stratified validation set, used for thresholds and tuning (`train_val_split` in `src/netad/data.py`).
- Preprocessing is fitted on the training part only: `log1p` + standard scaling on numeric columns, one-hot on `proto`, `service`, `state` with rare and unseen categories grouped together (`build_preprocessor` in `src/netad/preprocessing.py`).
- Test is not modified. Results are reported on the full test set, on the test rows whose features do not appear in train (10.4% of test rows do) and on the distinct test rows (55,945 of 82,332, each feature vector + category once, `distinct_rows` in `src/netad/data.py`).

### Known dataset issues
- The train/test overlap is uneven across classes: 49% of DoS test rows also appear in train, against 4% of Normal. DoS scores on the full test set will be optimistic.
- In test, 91% of Analysis and 90% of Backdoor rows share their feature vector with another class. Multi-class accuracy can't go above ~94%, and recall on these two classes can't go above ~10%. Binary detection is not affected.
- The TTL columns (`sttl`, `dttl`, `ct_state_ttl`) alone give ~79% test accuracy. They most likely reflect the testbed topology rather than attack behaviour.
- Test repeats some attack flows many times, mostly ones that are easy to detect (Generic is 42% of test attacks, 17% of the distinct ones). Counting each distinct flow once, test recall drops from 89% to 80% for the random forest and from 61% to 29% for Isolation Forest.

Details in `notebooks/01_eda.ipynb`.

## Evaluation
- Binary: precision, recall, F1, FPR, ROC-AUC, PR-AUC and confusion matrix. The alert threshold is chosen on validation with an FPR budget (1% by default), not on test.
- Detection rate per attack category, and `macro_detection`, their mean over the 9 attack categories. Recall depends on which categories are frequent in a given set, `macro_detection` does not.
- Multi-class: accuracy, macro and weighted F1, per-class precision/recall/F1 and confusion matrix.
- Every model is reported on validation, on the full test set, on the unseen test rows and on the distinct test rows. All rows give the alert volume on the traffic as it comes. Models are selected on validation and reported on the distinct test rows; test results never change which model is picked.
- Results are collected in `reports/results.json` (`src/netad/evaluation.py`).

## Notebooks
- `00_original_reference.ipynb` – the old version, kept for reference only
- `01_eda.ipynb` – dataset analysis
- `02_supervised.ipynb` – supervised models: tuning, test results, TTL ablation, permutation importance
- `03_anomaly.ipynb` – models trained on normal traffic only (Isolation Forest, autoencoder), compared with the supervised ones. The trained autoencoder, its preprocessor and threshold are saved to `models/` (not in git)
- `checks.ipynb` – asserts on `src/netad/`, run it after every change there
