# Project report

The full final report is available as the [original PDF](project-report/final-project-report.pdf). This page is an English orientation to the code currently in the repository; it is not a translation or replacement for the submitted report.

## Objective and workflow

The project studies supervised classification of chess-player skill bands using game-derived features. The preprocessing code calculates each game's average player rating and places records into three percentile-based target classes. It then samples the records, creates features from game length, opening depth, game outcome, time control, rated status, and ECO opening families, and splits the data into training, validation, and test sets.

Numeric scaling is fitted using training data. SMOTE is applied to the training split to address class imbalance. The project compares NumPy implementations of logistic regression and a feed-forward neural network with scikit-learn's RBF SVM and random forest. Evaluation utilities produce classification metrics, cross-validation results, confusion matrices, ROC plots, and model comparison reports.

## Reproducing the pipeline

From the repository root:

```bash
cd proyecto
python -m pip install -r requirements.txt
python main.py
```

Use `python main.py --help` for options such as `--skip-eda`, `--model`, and `--n-samples`. Generated evaluation outputs are written in `proyecto/`.

## Important consistency note

The current preprocessing implementation creates **three** target classes from rating percentiles. In contrast, some historical documentation and `production_predictor.py` describe or assume a **two-class** Beginner/Intermediate problem. Consequently, older result tables and the prediction helper should not be treated as a verified production interface for the current pipeline. Align the target labels, saved model artifacts, thresholds, and prediction API, then regenerate and validate the results before making performance claims.

The percentile cutoffs are calculated from the full source dataset before sampling and splitting. This defines labels relative to that dataset; results should not be interpreted as a universal measure of chess skill.

## Data and licensing

The source code is covered by the repository's MIT License. The dataset, pre-trained model files, and original academic documents can have separate ownership, licensing, and attribution terms. Verify those terms before redistributing the repository publicly.
