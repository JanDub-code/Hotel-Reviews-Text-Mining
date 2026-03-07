# Phase 5 summary

## Setup
- text column: `review_text`
- label scheme: `binary`
- TF-IDF bigrams enabled: `True`
- tested k values: `0, 5000, 10000, 20000, 30000`

## Best overall configuration
- best model: `linear_svm`
- best feature-selection k: `5000`
- macro F1: `0.8120`
- accuracy: `0.9503`

## Linear SVM comparison
- baseline macro F1 without feature selection: `0.8039`
- best macro F1 with feature selection: `0.8120` at `k=5000`
- delta macro F1: `+0.0082`
- delta accuracy: `+0.0023`

## Output files
- summary table: `outputs/feature_selection/feature_selection_summary.csv`
- linear SVM comparison: `outputs/feature_selection/feature_selection_linear_svm.csv`
- top features for the best configuration: `outputs/feature_selection/feature_selection_top_features.csv`
