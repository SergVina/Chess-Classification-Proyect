# Prediction helper: status and usage notes

`proyecto/production_predictor.py` contains a prediction helper that loads serialized models from `proyecto/models/`. It is retained as part of the project, but it should currently be treated as an experimental/legacy example rather than a production-ready API.

The current data pipeline creates three percentile-based target classes. The helper defines only two class names and uses binary decision thresholds. In addition, input feature encoding and scaling must match the training pipeline; the helper does not currently apply the complete preprocessing pipeline itself. The checked-in model files and older threshold values may therefore not be compatible with a freshly trained model.

Before using this helper for real predictions:

1. Choose and document the intended target classes.
2. Save the fitted preprocessing transforms alongside the model and apply them to incoming records.
3. Update prediction labels, probability handling, and thresholds for the chosen number of classes.
4. Retrain the models and validate the full training-to-inference path with representative, held-out examples.

This note intentionally avoids presenting example predictions as reliable until those issues are resolved.
