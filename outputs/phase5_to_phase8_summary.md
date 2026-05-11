# Phase 5 to 8 summary - final benchmark run

## Phase 5 - Feature selection and classification
- method: `SelectKBest(chi2)`
- text for classification: `review_text`
- label scheme: binary (`<= 5.0` negative, `>= 8.0` positive)
- representation: TF-IDF, unigram + bigram
- final full-dataset benchmark setting: `k=20000`

### Final result
- best final model: `Linear SVM`
- `macro_f1 = 0.8510`
- `accuracy = 0.9590`
- `weighted_f1 = 0.9566`

### Interpretation
- the final full-dataset run produced stronger classification results than the earlier 50k development experiments
- the negative class remains smaller and harder, so macro metrics are more informative than accuracy alone

### Top sentiment features in the final model
- positive: `excellent`, `amazing`, `lovely`, `fantastic`, `great`, `loved`, `perfect`, `definitely stay`
- negative: `dirty`, `worst`, `disinterested`, `filthy`, `unclean`, `impolite`, `rude`, `unhelpful`

## Phase 6 - Clustering
- corpus: `negative_text`
- sample: `50,000` rows
- after empty-text filtering and cleaning: `35,088` documents
- representation: TF-IDF, unigram + bigram
- final `k`: `7`

### Final result
- silhouette score: `0.0085`
- smallest cluster: `366` documents
- largest cluster: `19,397` documents

### Manual cluster interpretation for `k=7`
- `cluster_0`: generic dislike or short negative fragments
- `cluster_1`: room comfort, service, bathroom, and night issues
- `cluster_2`: Wi-Fi and air-conditioning problems
- `cluster_3`: broad miscellaneous hotel and staff comments
- `cluster_4`: small-room complaints
- `cluster_5`: breakfast quality, price, and inclusion
- `cluster_6`: parking and car-access costs

### Interpretation and limits
- clusters are interpretable but weakly separated
- the low silhouette score must be treated as a limitation, not hidden
- clustering is mainly an exploratory thematic summary, not a precise segmentation method

## Phase 7 - Information retrieval
- index: TF-IDF over `review_text`
- final run: full available dataset via `--sample-size 999999`
- similarity: cosine similarity
- final query: `dirty room and noisy street`

### Final top results
1. `very noisy dirty`
2. `room on the street side very noisy`
3. `room on the street very noisy`
4. `the room was very noisy from the street`
5. `street was noisy`

### Interpretation
- retrieval works well for concrete multi-word queries with explicit aspects
- broad and generic queries are less reliable without a relevance benchmark or more advanced ranking method

## Phase 8 - Visuals and outputs
Important generated artifacts:

- `outputs/classification_results.json`
- `outputs/classification_summary.csv`
- `outputs/clustering_results.json`
- `outputs/clustering_summary.csv`
- `outputs/retrieval_results_query.csv`
- `outputs/visuals/reviewer_score_distribution.png`
