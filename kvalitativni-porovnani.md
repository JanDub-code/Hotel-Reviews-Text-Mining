# Kvalitativní porovnání - zbývající neopravené body

## Verdikt po finálním spuštění

Pipeline se podařilo spustit nad dodaným datasetem. Klasifikace a retrieval běžely nad celým dostupným datasetem bez downsamplingu; clustering běžel podle plánu na 50,000 řádcích.

Hlavní výsledky finálního běhu:

- classification: Linear SVM, macro F1 = 0.8510, accuracy = 95.90 %
- clustering: `k = 7`, silhouette score = 0.0085
- retrieval query `dirty room and noisy street`: doběhla a vrátila relevantní top výsledky

Níže zůstávají pouze skutečné limity, ne chyby blokující odevzdání.

## 1. Dataset není součástí ZIPu

Dataset `Hotel_Reviews.csv` není přibalený v repozitáři. Je to prakticky správně kvůli velikosti a distribuci přes Kaggle, ale při kontrole je nutné postupovat podle `DATASET_LINKS.md` a stáhnout dataset ručně.

**Dopad:** nízký až střední. Projekt je reprodukovatelný, ale není samostatně spustitelný bez stažení dat.

## 2. Clustering má velmi nízké silhouette score

Finální clustering má `k = 7` a silhouette score `0.0085`, což znamená velmi slabou separaci clusterů. Report to přiznává a interpretuje clustering jako exploratorní nástroj.

**Dopad:** střední. Není to chyba implementace, spíš vlastnost dat: krátké, šumové a pseudo-negativní recenze tvoří neostré tematické hranice.

## 3. Retrieval je hodnocen hlavně kvalitativně

Retrieval modul má ukázkové dotazy a ruční kvalitativní posouzení, ale nemá anotovaný relevance benchmark, precision@k, recall@k ani MAP/nDCG.

**Dopad:** nízký až střední. Pro školní projekt je kvalitativní hodnocení pravděpodobně přijatelné, ale při obhajobě je vhodné uvést, že exaktnější IR evaluace by vyžadovala ručně anotované relevantní výsledky.

## 4. Použité metody jsou klasické, ne moderní embeddingové/transformerové

Projekt používá TF-IDF, klasické klasifikátory, K-Means a cosine similarity. Neobsahuje sentence embeddings, BERT/transformer baseline, BM25 ani NMF topic modeling.

**Dopad:** nízký. Zadání nevyžaduje nejmodernější metody a klasické metody dobře odpovídají výukovým tématům. Jde spíš o možnou otázku u obhajoby: proč nebyly použity embeddingy nebo transformerový model.

## 5. Clustering není spuštěný na celém datasetu

Klasifikace a retrieval byly spuštěny nad celým dostupným datasetem, ale clustering zůstal na 50,000 řádcích. To je metodicky obhajitelné kvůli rychlosti, paměti a interpretovatelnosti.

**Dopad:** nízký. V reportu je tento kompromis výslovně uvedený.
