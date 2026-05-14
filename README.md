# Text Mining Project

## Final working title
**Sentiment Classification, Topic Clustering and Similarity Search in Hotel Reviews**

Česky:
**Klasifikace sentimentu, shlukování témat a vyhledávání podobných hotelových recenzí**

## 1. Why this project fits the course
Projekt je navržený tak, aby přímo pokryl hlavní části sylabu předmětu Text Mining:

- **text representation and preprocessing** – čištění textu, tokenizace, n-gramy, TF-IDF
- **information retrieval** – vyhledávání podobných recenzí pomocí cosine similarity
- **text classification** – klasifikace sentimentu podle skóre recenze
- **text clustering** – shlukování negativních recenzí do tematických skupin
- **feature selection** – výběr atributů pomocí chi-square
- **optional language modeling overlap** – práce s n-gramy a porovnání reprezentací jako volitelný bonus

Projekt tak odpovídá požadavku na vlastní text-mining úlohu: definovat problém, získat data, data připravit, navrhnout řešení, vyhodnotit výsledky a interpretovat je.

## 2. Project goal
Cílem projektu je analyzovat hotelové recenze a ukázat kompletní text-mining pipeline na reálném datasetu.

Hlavní cíle:

1. převést textové recenze do strukturované reprezentace vhodné pro strojové učení,
2. natrénovat a porovnat modely pro klasifikaci sentimentu,
3. identifikovat hlavní témata v negativních recenzích pomocí shlukování,
4. realizovat jednoduché vyhledávání podobných recenzí,
5. zhodnotit vliv preprocessingu a výběru atributů na kvalitu výsledků.

## 3. Research questions
1. Jaký vliv má TF-IDF reprezentace a výběr atributů na kvalitu klasifikace sentimentu?
2. Které modely fungují na hotelových recenzích nejlépe: Naive Bayes, Logistic Regression nebo Linear SVM?
3. Jaká témata se objevují v negativních recenzích nejčastěji?
4. Lze pomocí TF-IDF a cosine similarity smysluplně vyhledávat podobné recenze?

## 4. Dataset
Primární dataset:
- Kaggle: `515K Hotel Reviews Data in Europe`
- Soubor: `Hotel_Reviews.csv`

### Practical size recommendation
Dataset má přibližně 515k řádků, ale pro školní projekt není nutné jako výchozí volba zpracovávat vše.

Doporučený režim práce:
- **rychlé ladění / development:** 20k--50k záznamů
- **reportované experimenty:** 20k--50k záznamů podle typu úlohy
- **celý dataset (~515k):** pouze jako volitelný finální benchmark nebo běh přes noc

Důvod: pro TF-IDF + klasické modely (`MultinomialNB`, `LogisticRegression`, `LinearSVC`) přináší vzorky 20k--50k dostatečně rychlou a stabilní základnu pro porovnání metod, zatímco plný dataset výrazně zvyšuje čas i paměťové nároky. Větší praktický dopad než samotné navyšování dat má u tohoto datasetu ošetření placeholderů typu `No Negative` / `No Positive` a nevyváženost tříd.

Alternativní mirror:
- Hugging Face: `enelpol/booking_com_reviews`

### Expected important columns
- `Positive_Review`
- `Negative_Review`
- `Reviewer_Score`
- případně další metadata jako `Hotel_Name`, `Tags`, `Average_Score`, `Reviewer_Nationality`

### Verified dataset snapshot
Lokálně ověřený stav souboru `data/raw/Hotel_Reviews.csv` k 2026-03-07:

- `515,738` řádků a `17` sloupců
- načtení bez chyb přes `pandas.read_csv(...)` i přes projektový loader `src.common.load_dataset(...)`
- klíčové datové typy: `Reviewer_Score` a `Average_Score` jsou `float64`, textová pole jsou `object`
- chybějící hodnoty: `Positive_Review` `0`, `Negative_Review` `0`, `Reviewer_Score` `0`, `lat` `3,268`, `lng` `3,268`
- `Reviewer_Score`: min `2.5`, medián `8.8`, průměr `8.3951`, max `10.0`
- placeholdery k ošetření v preprocessingu: `152,663` hodnot typu `No Negative`, `37,791` hodnot typu `No Positive`
- délka recenzí: `Negative_Review` průměr `17.85` slov, 95. percentil `67`; `Positive_Review` průměr `16.47` slov, 95. percentil `54`

Tento snapshot je vhodný jako podklad do sekce dataset/EDA v reportu. Finální reportový text ale stále patří do `main.tex`.

## 4.1 Environment baseline
Repo je nyní sjednocený na tomto výchozím prostředí:

- Python `3.14.3`
- přesná sada balíčků v `requirements-lock.txt`
- minimální kompatibilní verze ponechané v `requirements.txt`

Doporučení pro tým:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 5. Proposed task design
### 5.1 Sentiment classification
Vytvoří se textový atribut například:

```text
review_text = Positive_Review + " " + Negative_Review
```

Možné značení tříd:

### Varianta A: binární klasifikace
- `Reviewer_Score <= 5.0` -> negative
- `Reviewer_Score >= 8.0` -> positive
- střední hodnocení se vynechají

Tato varianta bývá nejstabilnější pro první běh.

### Varianta B: třída negative / neutral / positive
- `Reviewer_Score <= 5.0` -> negative
- `5.0 < Reviewer_Score < 8.0` -> neutral
- `Reviewer_Score >= 8.0` -> positive

### 5.2 Topic clustering
Pro shlukování se doporučuje použít hlavně pole `Negative_Review`, protože se témata problémů interpretují lépe než na smíšeném textu.

Předpokládaná témata:
- čistota pokoje
- hluk
- personál
- koupelna
- snídaně
- lokalita
- cena / hodnota

### 5.3 Similarity search
Zadaný dotaz nebo konkrétní recenze se převede do TF-IDF prostoru a vrátí se nejpodobnější recenze pomocí cosine similarity.

## 6. Methodology
### 6.1 Data preparation
- odstranění prázdných a zjevně neinformatických záznamů
- spojení pozitivní a negativní části recenze
- základní čištění textu
- lowercasing
- odstranění nadbytečných mezer
- odstranění speciálních znaků dle potřeby

### 6.2 Text representation
- unigramy a bigramy
- Bag of Words / TF-IDF
- omezení slovníku pomocí `min_df`, `max_df`, `max_features`

### 6.3 Feature selection
- `SelectKBest(chi2)`
- porovnání výkonu modelu bez a s výběrem atributů

### 6.4 Classification models
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

### 6.5 Clustering
- K-Means nad TF-IDF reprezentací
- interpretace clusterů pomocí top terms a reprezentativních recenzí
- volitelně silhouette score

### 6.6 Information retrieval
- TF-IDF
- cosine similarity
- top-k nejpodobnějších dokumentů

## 7. Evaluation
### Klasifikace
- accuracy
- precision
- recall
- F1-score
- confusion matrix

### Shlukování
- silhouette score
- top terms v clusterech
- ruční interpretace reprezentativních recenzí

### Vyhledávání
- kvalitativní posouzení relevance výsledků
- několik ukázkových dotazů / referenčních recenzí

## 8. Expected outputs
- porovnání klasifikačních modelů
- tabulka výsledků pro různé konfigurace
- seznam nejdůležitějších slov pro pozitivní a negativní sentiment
- interpretace clusterů negativních recenzí
- ukázky podobných recenzí vracených retrieval modulem

## 9. Suggested report structure
1. Introduction and motivation
2. Project goal
3. Dataset description
4. Data preprocessing
5. Text representation
6. Classification experiments
7. Feature selection experiments
8. Clustering experiments
9. Similarity search module
10. Discussion of results
11. Conclusion

## 9.1 Report writing convention
- `REPORT_TEMPLATE.md` slouží jen jako obsahová osnova.
- Finální text reportu se píše do `main.tex`.
- `main.tex` je hlavní odevzdávaný LaTeX dokument.
- Strukturu z `REPORT_TEMPLATE.md` používat jako vodítko, ale jednotlivé kapitoly průběžně psát a upravovat přímo v `main.tex`.
- Pokud je v `main.tex` starý nebo nesouvisející obsah, bere se jako placeholder k nahrazení textem k projektu hotelových recenzí.

## 10. Repository structure
```text
text_mining_hotel_reviews_project/
├── .python-version
├── README.md
├── TODO.md
├── DATASET_LINKS.md
├── REPORT_TEMPLATE.md
├── main.tex
├── requirements.txt
├── requirements-lock.txt
├── data/
│   └── raw/
├── outputs/
└── src/
    ├── common.py
    ├── classification.py
    ├── clustering.py
    ├── retrieval.py
    └── run_all.py
```

## 11. Quick start
1. stáhnout dataset podle `DATASET_LINKS.md`
2. uložit `Hotel_Reviews.csv` do `data/raw/`
3. vytvořit prostředí a nainstalovat balíčky:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

4. spustit klasifikaci:

```bash
python src/classification.py --data data/raw/Hotel_Reviews.csv --full-dataset --use-bigrams --feature-selection-k 20000
```

5. spustit clustering:

```bash
python src/clustering.py --data data/raw/Hotel_Reviews.csv --sample-size 50000 --k 7 --use-bigrams
```

6. spustit retrieval:

```bash
python src/retrieval.py --data data/raw/Hotel_Reviews.csv --full-dataset --query "dirty room and noisy street" --top-k 5
```

Poznámka: Přepínač `--full-dataset` vypíná vzorkování. Klasifikace a retrieval jsou nastavené na celý dostupný dataset; clustering zůstává na 50k kvůli rychlosti, paměti a interpretovatelnosti.

## 12. Recommended final scope
### Core scope
- TF-IDF reprezentace
- feature selection
- sentiment classification
- clustering negativních recenzí
- jednoduchý IR modul

### Optional bonus
- porovnání unigram vs. unigram+bigram
- NMF topic modeling jako bonusová alternativa ke clusteringu
- sentence embeddings / transformer baseline

## 13. Practical workflow for the report
Doporučený postup:

1. použít `REPORT_TEMPLATE.md` jako checklist kapitol,
2. průběžné výsledky a interpretace zapisovat rovnou do `main.tex`,
3. tabulky, grafy a ukázky výstupů ukládat do `outputs/`,
4. před finálním odevzdáním zkontrolovat, že `README.md`, `TODO.md` a `main.tex` jsou obsahově konzistentní.
