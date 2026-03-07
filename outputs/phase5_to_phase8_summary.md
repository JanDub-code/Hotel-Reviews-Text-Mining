# Phase 5 to 8 summary

## Phase 5 — Feature selection
- metoda: `SelectKBest(chi2)`
- text pro klasifikaci: `review_text`
- label scheme: binární (`<= 5.0` negative, `>= 8.0` positive)
- reprezentace: TF-IDF, unigram + bigram
- testované hodnoty `k`: `0`, `5000`, `10000`, `20000`, `30000`

### Výsledek
- nejlepší konfigurace: `Linear SVM` + `k=5000`
- `macro_f1 = 0.8120`
- `accuracy = 0.9503`
- baseline bez feature selection: `macro_f1 = 0.8039`, `accuracy = 0.9481`
- zlepšení proti baseline: `+0.0082` macro F1 a `+0.0023` accuracy

### Interpretace
- feature selection zde není jen kosmetická; při `k=5000` zlepšila `Linear SVM` i `MultinomialNB`
- `k=10000` už je slabší než `k=5000`
- `k=20000` a `k=30000` se vrací prakticky k baseline, protože skutečný počet aktivních feature byl jen `15,266`
- finální konfigurace pro report: `review_text` + TF-IDF bigramy + `SelectKBest(chi2, k=5000)` + `Linear SVM`

### Top sentiment features
- pozitivní: `great`, `lovely`, `fantastic`, `loved`, `perfect`, `amazing`, `good value`, `friendly helpful`
- negativní: `dirty`, `worst`, `shabby`, `poor`, `tired`, `unhelpful`, `rude`, `couldn sleep`

## Phase 6 — Clustering
- korpus: `negative_text`
- development sample: `20,000` recenzí
- po vyřazení prázdných textů a po cleaningu zůstalo `14,069` dokumentů
- reprezentace: TF-IDF, unigram + bigram
- testované hodnoty `k`: `5`, `6`, `7`, `8`, `9`

### Výsledek
- nejlepší konfigurace podle silhouette score: `k=7`
- silhouette score: `0.0100`

### Ručně pojmenované clustery pro `k=7`
- `cluster_0`: misc short or mixed complaints
- `cluster_1`: very small room
- `cluster_2`: breakfast quality and inclusion
- `cluster_3`: small outdated rooms
- `cluster_4`: expensive extras
- `cluster_5`: room comfort and maintenance
- `cluster_6`: staff and reception problems

### Interpretace a limity
- clustery jsou interpretovatelné, ale separace je slabá; nízké silhouette score je potřeba v reportu explicitně přiznat
- v datech zůstávají krátké nebo pseudo-negativní fráze typu `No failings`, `No complaints`, `Everything was perfect`, `Breakfast`, `Expensive`
- tyto krátké vstupy zhoršují čistotu clusterů a vytvářejí jeden výrazně „misc“ cluster
- clustering proto slouží hlavně jako exploratorní tematické shrnutí problémů, ne jako ostrá segmentace recenzí

## Phase 7 — Information retrieval
- index: TF-IDF nad `review_text`
- sample pro retrieval: `20,000` recenzí
- po cleaningu: `19,985` dokumentů
- matice indexu: `19,985 x 21,242`
- similarity: cosine similarity
- testováno `6` ukázkových dotazů

### Kvalitativní zhodnocení relevance
- silné dotazy: `friendly staff and great location`, `small bathroom and old furniture`, `excellent breakfast and comfortable bed`
- smíšené dotazy: `dirty room and noisy street`, `slow wifi and poor service`
- slabý dotaz: `good hotel`

### Interpretace
- retrieval funguje velmi dobře u konkrétních víceslovných dotazů s jasnými klíčovými termy
- široké a neurčité dotazy vracejí generické boilerplate texty a jsou méně užitečné
- systém lépe zachytí přesně pojmenované atributy než obecnou „kvalitu hotelu“

## Phase 8 — Visuals and outputs
Vygenerované artefakty:

- `outputs/visuals/reviewer_score_distribution.png`
- `outputs/visuals/model_results_table.csv`
- `outputs/visuals/top_features_table.csv`
- `outputs/visuals/cluster_summary_table.csv`
- `outputs/visuals/retrieval_examples_table.csv`

Další podklady:

- `outputs/feature_selection/phase5_summary.md`
- `outputs/clustering/phase6_summary.md`
- `outputs/retrieval/phase7_summary.md`
