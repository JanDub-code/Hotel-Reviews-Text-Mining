# Phase 3 and 4 summary - final benchmark run

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
- feature selection: `SelectKBest(chi2, k=20000)`

## Cleaning pipeline
1. replace placeholders in `Positive_Review` and `Negative_Review`
2. build `review_text`, `negative_text`, `positive_text`
3. drop empty selected text field
4. lowercase text
5. remove non-letter characters
6. normalize whitespace
7. drop empty cleaned texts

## Verified dataset flow on the full benchmark run
- rows loaded: `515,738`
- rows after binary label filtering: `366,349`
- rows after empty selected-text filtering: `365,997`
- rows after text cleaning: `365,988`
- train/test split: `292,790 / 73,198`
- label distribution after filtering: `335,365` positive, `30,623` negative

## Classification results

| Model | Macro precision | Macro recall | Macro F1 | Weighted F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | 0.8977 | 0.8158 | 0.8510 | 0.9566 | 0.9590 |
| Logistic Regression | 0.9131 | 0.7938 | 0.8412 | 0.9547 | 0.9582 |
| Multinomial Naive Bayes | 0.9044 | 0.7726 | 0.8230 | 0.9500 | 0.9544 |

## Decision
- keep `review_text` for sentiment classification
- keep unigram + bigram TF-IDF
- keep `SelectKBest(chi2, k=20000)` for the final full-dataset benchmark
- keep `Linear SVM` as the strongest final classifier
- keep `negative_text` mainly for clustering

## Interpretation
- `Linear SVM` is the best classifier by macro F1
- all three models benefit from the much larger final benchmark set compared with the earlier development sample
- class imbalance remains important: accuracy is high, but macro F1 is still the safer headline metric
