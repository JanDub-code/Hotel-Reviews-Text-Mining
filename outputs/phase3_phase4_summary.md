# Phase 3 and 4 summary

## Final preprocessing choice for classification
- text column: `review_text`
- label scheme: binary
- labels: `Reviewer_Score <= 5.0 -> negative`, `Reviewer_Score >= 8.0 -> positive`
- vectorizer: TF-IDF
- n-grams: unigram + bigram
- `min_df=5`
- `max_df=0.9`
- `max_features=50000`
- stop words: English stop-word list from scikit-learn

## Cleaning pipeline
1. replace placeholders in `Positive_Review` and `Negative_Review`
2. build `review_text`, `negative_text`, `positive_text`
3. drop empty selected text field
4. lowercase text
5. remove non-letter characters
6. normalize whitespace
7. drop empty cleaned texts

## Verified dataset flow on the 50k development sample
- sampled rows loaded: `50,000`
- rows after binary label filtering: `35,553`
- rows after text cleaning: `35,519`
- dropped by label filtering: `14,447`
- dropped by empty selected text: `33`
- dropped by empty cleaned text: `1`
- train/test split: `28,415 / 7,104`
- label distribution after filtering: `32,502` positive, `3,017` negative

## Classification results

### Unigram baseline
| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.8694 | 0.7592 | 0.8025 | 0.9431 | 0.9476 |
| Logistic Regression | 0.9117 | 0.7103 | 0.7721 | 0.9374 | 0.9461 |
| Multinomial Naive Bayes | 0.9322 | 0.6120 | 0.6639 | 0.9135 | 0.9329 |

### Unigram + bigram
| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.8717 | 0.7602 | 0.8039 | 0.9435 | 0.9481 |
| Logistic Regression | 0.9152 | 0.6982 | 0.7612 | 0.9350 | 0.9447 |
| Multinomial Naive Bayes | 0.9476 | 0.5926 | 0.6374 | 0.9080 | 0.9303 |

## Decision
- keep `review_text` for sentiment classification
- keep unigram + bigram TF-IDF as the current best configuration
- keep `Linear SVM` as the strongest baseline model
- keep `negative_text` as a separate variant mainly for clustering and targeted experiments

## First interpretation
- `Linear SVM` is the best classifier on both tested n-gram settings
- bigrams help only slightly, but they still improve the best result, so they stay in the baseline
- `Logistic Regression` stays competitive, but is consistently behind `Linear SVM`
- `Multinomial Naive Bayes` has high headline precision and accuracy, but weaker macro recall/F1, so it is less robust on the minority negative class
- the main bottleneck is class imbalance after binary labeling, not vectorizer capacity
- `negative_text` alone works, but its smoke-test performance is weaker than `review_text`, so it should not replace the full review for sentiment classification

## Key artifacts
- `outputs/classification_unigram/`
- `outputs/classification_bigram/`
- `outputs/classification_negative_only_smoke/`

Each run contains:
- `classification_results.json`
- `classification_summary.csv`
- `preprocessing_summary.json`
- `confusion_matrices.csv`
