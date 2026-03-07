# Dataset links

## Primary dataset
Kaggle:
https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe

Typical file:
- `Hotel_Reviews.csv`

## Alternative mirror
Hugging Face:
https://huggingface.co/datasets/enelpol/booking_com_reviews

## Kaggle API download example
Po nastavení Kaggle API tokenu lze použít:

```bash
kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe -p data/raw --unzip
```

## Manual download workflow
1. otevřít Kaggle dataset,
2. stáhnout zip,
3. rozbalit obsah,
4. umístit `Hotel_Reviews.csv` do složky `data/raw/`.

## Recommended local path
```text
data/raw/Hotel_Reviews.csv
```

## Notes
- Kaggle verze je vhodná jako hlavní zdroj.
- Hugging Face mirror je dobrý záložní zdroj, pokud budete chtít rychlý přístup k textovým polím.
- Pro tento projekt je nejlepší pracovat s originálním CSV, protože obsahuje jak textové sloupce, tak skóre a metadata.
