<div align="center">

# Chess Skill Classification

**A reproducible machine-learning study of chess-game features and player skill levels.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

This project explores whether features extracted from chess games can distinguish player skill levels. It compares classical machine-learning models with implementations built from scratch, investigates class imbalance, and produces evaluation reports and plots.

> **Scope note:** The current preprocessing code creates three rating-based skill bands (Beginner, Intermediate, Advanced). The archived academic report and the older production prediction helper still assume a two-class Beginner/Intermediate problem. Those legacy artifacts are not guaranteed to match a fresh run; see [known limitations](docs/PROJECT_REPORT.md).

## What the project does

The pipeline loads chess-game records, defines skill bands from the 33rd and 66th percentiles of average player rating, samples the data, and extracts game-level features. It then splits the data into training, validation, and test sets, fits numeric scalers on training data only, applies SMOTE to the training split, trains and evaluates models, and writes comparison reports and visualizations.

The feature set includes game length, opening depth, victory status, winner, rated status, time control, and ECO opening-family indicators. Player ratings are used to define the target, not included as input features.

Models included:

- Logistic regression and a feed-forward neural network implemented with NumPy.
- RBF-kernel support vector machine and random forest models using scikit-learn.
- An ensemble evaluation, stratified cross-validation, classification metrics, confusion matrices, and threshold analysis.

## Quick start

The application and its data live in `proyecto/`; run commands from that directory:

```bash
cd proyecto
python -m venv .venv
```

Activate the virtual environment, then install the dependencies and run the full pipeline:

```bash
python -m pip install -r requirements.txt
python main.py
```

To skip exploratory plots or train a single model:

```bash
python main.py --skip-eda
python main.py --skip-eda --model rf
python main.py --skip-eda --model lr
python main.py --skip-eda --model nn
python main.py --skip-eda --model svm
```

The default sample size is 2,500 games. Use `python main.py --help` to see the available options. The pipeline expects `proyecto/data/games.csv` and writes generated plots, model files, configuration, and evaluation reports under `proyecto/`.

## Results and reproducibility

Evaluation results depend on the selected sample size, the dataset version, and the current code. Reports and plots are generated locally and are not kept as authoritative, current results in the repository; rerun the pipeline to evaluate the current source. The accompanying [project report](docs/project-report/final-project-report.pdf) is the archived academic write-up.

## Repository guide

| Path | Contents |
| --- | --- |
| `proyecto/main.py` | Training and evaluation pipeline |
| `proyecto/preprocessing/` | Feature engineering, splitting, scaling, and sampling |
| `proyecto/models/` | Model implementations |
| `proyecto/evaluation/` | Metrics, plots, and report generation |
| `proyecto/data/games.csv` | Dataset used by the pipeline |
| `docs/project-report/` | Final project report |
| `docs/proposal/` | Original project proposal documents |
| `docs/PRODUCTION_GUIDE.md` | Status and limitations of the prediction helper |
| `docs/SMOTE.md` | Notes on the imbalance-handling step |

## Documentation

- [Project report and implementation notes](docs/PROJECT_REPORT.md)
- [Production prediction helper: current status](docs/PRODUCTION_GUIDE.md)
- [SMOTE notes](docs/SMOTE.md)
- [Archived proposal documents](docs/proposal/README.md)

## Limitations

- The target is derived from rating percentiles across the full input dataset, so the labels describe relative rating bands in that dataset rather than a universal chess title or rating scale.
- Some historical reports and `production_predictor.py` use a two-class interface, while the current preprocessing pipeline produces three target classes. Reconcile these before relying on the saved models or prediction helper.
- The dataset and serialized model files may have terms or provenance separate from this repository's MIT-licensed source code. Check their rights and attribution requirements before redistributing them.

## License

The project source code is released under the [MIT License](LICENSE). The license does not automatically apply to third-party datasets, model artifacts, or the archived academic documents.
