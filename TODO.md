# TODO plan

## Phase 0 — Administrative setup
- [x] potvrdit finální název projektu
- [ ] rozdělit roli v týmu (data, modeling, report, prezentace)
- [x] založit repozitář / sdílenou složku
- [x] sjednotit verzi Pythonu a balíčků

Status k 2026-03-07:
- finální pracovní název je sjednocený v `README.md`
- sdílená projektová struktura existuje lokálně v tomto repu
- baseline prostředí je zapsaný do `.python-version` a `requirements-lock.txt`

## Phase 1 — Dataset acquisition
- [x] stáhnout Kaggle dataset `515K Hotel Reviews Data in Europe`
- [x] ověřit, že soubor `Hotel_Reviews.csv` jde načíst bez chyb
- [x] zkontrolovat sloupce a datové typy
- [x] potvrdit pracovní režim: development 20k--50k, finální experimenty cca 100k, full dataset jen volitelně
- [x] uložit krátký dataset summary do reportu

Status k 2026-03-07:
- `data/raw/Hotel_Reviews.csv` je přítomný a byl úspěšně načten
- ověřený snapshot datasetu je zapsaný do `README.md`
- dataset summary je přenesený i do `main.tex`

## Phase 2 — Data understanding and EDA
- [x] spočítat počet záznamů
- [x] spočítat počty chybějících hodnot v textových sloupcích
- [x] prozkoumat rozdělení `Reviewer_Score`
- [x] najít časté placeholdery typu `No Negative`, `No Positive`
- [x] poznamenat do reportu, že dataset má ~515k řádků, ale pro hlavní experimenty stačí cca 100k
- [x] ověřit délku recenzí a případné outliery
- [x] rozhodnout finální label scheme

Status k 2026-03-07:
- počet záznamů: `515,738`
- textová pole `Positive_Review` a `Negative_Review` nemají chybějící hodnoty
- `Reviewer_Score` má rozsah `2.5` až `10.0`, medián `8.8` a průměr `8.3951`
- placeholdery: `152,663` hodnot typu `No Negative`, `37,791` hodnot typu `No Positive`
- délka recenzí je zkontrolovaná přes průměr, 95. percentil a maximum; extrémy existují, ale největší praktický problém zůstávají placeholdery
- finální label scheme pro klasifikaci je zatím binární: `<= 5.0` negative, `>= 8.0` positive

## Phase 3 — Preprocessing
- [x] vytvořit společný text `review_text`
- [x] vytvořit variantu pouze pro `Negative_Review`
- [x] odstranit prázdné a neinformatické záznamy
- [x] otestovat unigramy
- [x] otestovat unigramy + bigramy
- [x] nastavit `min_df`, `max_df`, `max_features`
- [x] dokumentovat finální preprocessing pipeline

Status k 2026-03-07:
- `src/common.py` vytváří `review_text`, `negative_text` a `positive_text`
- placeholders jsou nahrazené prázdným textem a pipeline zahazuje prázdné vstupy před i po cleaning kroku
- otestované konfigurace jsou uložené v `outputs/classification_unigram/` a `outputs/classification_bigram/`
- finální baseline pro klasifikaci: `review_text` + TF-IDF + unigramy/bigramy + `min_df=5`, `max_df=0.9`, `max_features=50000`
- dokumentace a interpretace jsou shrnuté v `outputs/phase3_phase4_summary.md`

## Phase 4 — Sentiment classification
- [x] připravit binární labely z `Reviewer_Score`
- [x] udělat train/test split se stratifikací
- [x] natrénovat Multinomial Naive Bayes
- [x] natrénovat Logistic Regression
- [x] natrénovat Linear SVM
- [x] porovnat accuracy, precision, recall, F1
- [x] uložit confusion matrix
- [x] sepsat první interpretaci výsledků

Status k 2026-03-07:
- klasifikační běhy jsou uložené v `outputs/classification_unigram/` a `outputs/classification_bigram/`
- nejlepší baseline je `Linear SVM` s TF-IDF unigram+bigram, `macro_f1=0.8039`, `accuracy=0.9481`
- confusion matrices jsou uložené samostatně v `confusion_matrices.csv`
- první interpretace výsledků je zapsaná v `outputs/phase3_phase4_summary.md`

## Phase 5 — Feature selection
- [x] aplikovat `SelectKBest(chi2)`
- [x] otestovat více hodnot `k`
- [x] porovnat výkon před a po feature selection
- [x] vytáhnout nejdůležitější termy pro sentiment
- [x] rozhodnout finální konfiguraci pro report

Status k 2026-03-07:
- testované hodnoty `k`: `0`, `5000`, `10000`, `20000`, `30000`
- nejlepší konfigurace je `Linear SVM` + `SelectKBest(chi2, k=5000)`
- `macro_f1` se zlepšilo z `0.8039` na `0.8120`
- top sentiment features jsou uložené v `outputs/feature_selection/feature_selection_top_features.csv`
- shrnutí je v `outputs/feature_selection/phase5_summary.md` a `outputs/phase5_to_phase8_summary.md`

## Phase 6 — Clustering
- [x] připravit korpus z negativních recenzí
- [x] spustit K-Means pro více hodnot `k`
- [x] spočítat silhouette score
- [x] vytáhnout top terms v každém clusteru
- [x] najít reprezentativní recenze pro každý cluster
- [x] ručně pojmenovat clustery
- [x] popsat limity clusteringu

Status k 2026-03-07:
- clustering běžel nad `negative_text` na sample `20,000`, po cleaningu `14,069` dokumentů
- testované hodnoty `k`: `5`, `6`, `7`, `8`, `9`
- nejlepší `k` podle silhouette score je `7` s hodnotou `0.0100`
- ruční jména clusterů jsou v `outputs/clustering/k_7/cluster_labels.csv`
- shrnutí clusterů a limity jsou v `outputs/clustering/phase6_summary.md` a `outputs/phase5_to_phase8_summary.md`

## Phase 7 — Information retrieval
- [x] vytvořit TF-IDF index nad recenzemi
- [x] připravit 5 až 10 ukázkových dotazů
- [x] otestovat cosine similarity
- [x] vybrat reprezentativní příklady dobrých a slabších výsledků
- [x] sepsat krátké zhodnocení relevance výsledků

Status k 2026-03-07:
- retrieval index je uložený v `outputs/retrieval/tfidf_index.npz`
- index byl vytvořen nad `19,985` recenzemi a má rozměr `19,985 x 21,242`
- otestováno bylo `6` dotazů
- kvalitativní hodnocení dotazů je v `outputs/retrieval/query_assessment.csv`
- shrnutí relevance je v `outputs/retrieval/phase7_summary.md` a `outputs/phase5_to_phase8_summary.md`

## Phase 8 — Visuals and outputs
- [x] graf rozdělení skóre recenzí
- [x] tabulka s výsledky modelů
- [x] tabulka top features
- [x] shrnutí clusterů
- [x] ukázka retrieval výsledků

Status k 2026-03-07:
- graf skóre je v `outputs/visuals/reviewer_score_distribution.png`
- tabulka modelů je v `outputs/visuals/model_results_table.csv`
- tabulka top features je v `outputs/visuals/top_features_table.csv`
- shrnutí clusterů je v `outputs/visuals/cluster_summary_table.csv`
- retrieval ukázky jsou v `outputs/visuals/retrieval_examples_table.csv`

## Phase 9 — Report writing
- [x] brát `REPORT_TEMPLATE.md` pouze jako osnovu, ne jako finální report
- [x] psát finální report přímo do `main.tex`
- [x] nahradit případný placeholder / starý obsah v `main.tex` textem k hotelovým recenzím
- [x] promítnout strukturu z `REPORT_TEMPLATE.md` do kapitol v `main.tex`
- [x] dopsat úvod a motivaci
- [x] popsat dataset
- [x] popsat preprocessing
- [x] popsat klasifikační experimenty
- [x] popsat clustering
- [x] popsat retrieval část
- [x] doplnit diskusi výsledků
- [x] doplnit limity a future work
- [x] sjednotit styl, citace a formátování

Status k 2026-03-07:
- `main.tex` je kompletně přepsaný na projekt hotelových recenzí podle struktury z `REPORT_TEMPLATE.md`
- starý placeholder obsah o akciových trzích byl odstraněn
- report využívá výsledky z Phase 1 až 8 a odkazuje na vygenerované artefakty v `outputs/visuals/`
- `main.pdf` byl úspěšně zkompilován pomocí `xelatex`
- pro lokální build byly doplněny kompatibilní shim soubory `mathspec.sty` a `xltxtra.sty`, protože je aktuální TeX instalace neobsahovala

## Phase 10 — Defense preparation
- [ ] připravit 5min shrnutí projektu
- [ ] umět zdůvodnit výběr reprezentace textu
- [ ] umět vysvětlit rozdíl mezi klasifikací a clusteringem
- [ ] umět vysvětlit TF-IDF a cosine similarity
- [ ] umět vysvětlit chi-square feature selection
- [ ] umět okomentovat chyby modelu a limity datasetu

## Minimum viable project
Když bude málo času, minimum pro solidní odevzdání je:
- [x] TF-IDF reprezentace
- [x] binární sentiment klasifikace
- [x] porovnání 3 modelů
- [x] feature selection experiment
- [x] clustering negativních recenzí
- [x] základní interpretace výsledků
