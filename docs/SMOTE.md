# SMOTE

The training pipeline uses SMOTE (Synthetic Minority Over-sampling Technique) to increase representation of minority classes by interpolating between training examples. In this project it is applied after the train/validation/test split and only to the training data; validation and test distributions remain unchanged for evaluation.

SMOTE is one experiment in the pipeline, not a guarantee of improved performance. Compare metrics on untouched validation/test data, and consider whether synthetic interpolation is appropriate for the mixed numeric and one-hot encoded features used here.

The dependency is included in `proyecto/requirements.txt`. The implementation is in `proyecto/preprocessing/smote.py`.
