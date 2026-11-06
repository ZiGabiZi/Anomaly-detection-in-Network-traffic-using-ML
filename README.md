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
- ✅ Anomaly detection: leave-one-attack-out
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
- The TTL columns (`sttl`, `dttl`, `ct_state_ttl`) alone give ~79% test accuracy. They most likely reflect the testbed topology rather than attack behaviour. In training, `sttl` 31 appears only in normal traffic, attacks have 62 or 254. Normal test flows with sttl 31 get no false alarms from the random forest, those with 254 get 8.5%, and test has twice as much of that traffic as validation (41% vs 19% of normal flows). That accounts for the whole rise from 1% FPR on validation to 4.0% on test. Isolation Forest and the autoencoder show the same pattern without ever seeing an attack.
- Test repeats some attack flows many times, mostly ones that are easy to detect (Generic is 42% of test attacks, 17% of the distinct ones). Counting each distinct flow once, test recall drops from 89% to 80% for the random forest and from 61% to 29% for Isolation Forest.

Details in `notebooks/01_eda.ipynb`.

## Evaluation
- Binary: precision, recall, F1, FPR, ROC-AUC, PR-AUC and confusion matrix. The alert threshold is chosen on validation with an FPR budget (1% by default), not on test (`choose_threshold` in `src/netad/evaluation.py`). The supervised models use the `"fpr"` mode: highest recall within the budget, so attack labels of validation are used too. Isolation Forest and the autoencoder use `"normal_quantile"`: the lowest normal score with at most 1% of normal validation flows at or above it, so no attack example is needed. Switching the anomaly detectors from `"fpr"` to `"normal_quantile"` left test results practically identical (recall changed by under 0.1 point).
- Detection rate per attack category, and `macro_detection`, their mean over the 9 attack categories. Recall depends on which categories are frequent in a given set, `macro_detection` does not.
- Multi-class: accuracy, macro and weighted F1, per-class precision/recall/F1 and confusion matrix.
- Every model is reported on validation, on the full test set, on the unseen test rows and on the distinct test rows. All rows give the alert volume on the traffic as it comes. Final model selection uses validation metrics. Distinct test rows are used for exploratory comparisons.
- Distinct test rows keep one row per pair of feature vector and `attack_cat`. Exact repeats within a category count once. If a vector has multiple category labels, each label is retained. Deduplication does not establish statistical independence: 8.7% of them differ from another row only in the 11 timing columns (`dur`, `rate`, `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`), a third of Generic, and 4.6% share their feature vector with another category (12 rows have both Normal and an attack label, the rest are attack vs attack). The distinct subset shows how detection rates change when each feature vector and category pair has equal weight. Use the full test set for operational comparisons of FPR, precision and alert volume, where each flagged row counts as an alert. Grouping flows properly would need source IPs and timestamps, which only the full UNSW-NB15 CSVs have; for the same reason a bootstrap on distinct rows would give intervals that are too narrow.
- Results are collected in `reports/results.json` (`src/netad/evaluation.py`).
- Anomaly detector settings are compared with seeds 42, 1 and 2. The final evaluation uses seed 42. Autoencoder results can vary across machines and library versions, including the epoch at which training stops.

## Evaluation protocol
The UNSW-NB15 test split was inspected repeatedly during development. Its results are exploratory because that inspection informed choices about preprocessing, reporting and follow-up experiments.

Decisions based on training and validation data
- hyperparameters of the supervised models (cross-validation on the training part);
- the final model of each task (validation partial ROC-AUC / macro-F1; the random forest was kept although gradient boosting scored higher on test);
- Isolation Forest and autoencoder settings (validation, 3 seeds);
- alert thresholds under a 1% validation FPR budget (highest recall within the budget for the supervised models, normal validation scores only for Isolation Forest and the autoencoder).

Decisions informed by test inspection
- reporting on distinct test rows and adding `macro_detection`, after seeing how repeated flows raise recall;
- the six categories held out in `04_leave_one_out`, with exclusions informed in part by earlier test results;
- the FPR breakdown by `sttl`, computed on test;
- `OneHotEncoder(handle_unknown="infrequent_if_exist")`, after seeing `state` values in test that train doesn't have (5 rows).

The current configurations are fixed for this benchmark. Any future model development uses training and validation data. Before evaluating on independently collected traffic, freeze the preprocessing, model, threshold and metrics. Use that evaluation to assess generalisation beyond the UNSW-NB15 testbed. CIC-UNSW-NB15 and the full 2.5M-record UNSW-NB15 come from the same captures and testbed, so they don't qualify; a model trained on CICFlowMeter features (CIC-UNSW-NB15) and evaluated on independently collected traffic with compatible feature definitions is one possible design.

## Notebooks
- `00_original_reference.ipynb` – the old version, kept for reference only
- `01_eda.ipynb` – dataset analysis
- `02_supervised.ipynb` – supervised models: tuning, test results, TTL ablation, permutation importance
- `03_anomaly.ipynb` – models trained on normal traffic only (Isolation Forest, autoencoder), compared with the supervised ones. The trained autoencoder, its preprocessor and threshold are saved to `models/` (not in git)
- `04_leave_one_out.ipynb` – each attack category held out of training in turn: does the random forest, Isolation Forest or the autoencoder still catch it
- `checks.ipynb` – asserts on `src/netad/`, run it after every change there
