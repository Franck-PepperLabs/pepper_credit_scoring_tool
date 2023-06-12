# Data Project Steps

This project follows a structured approach, organized into different phases within the `notebooks/` directory. Each phase represents a specific stage in the data analysis and modeling process.

By following this systematic workflow, the project progresses through each phase, covering essential steps from data acquisition to advanced modeling and evaluation. Each notebook within the respective phase folder provides a clear progression in the project's data-driven journey.

## Phase 1: Data Acquisition and Preprocessing

The first phase, preceding the Exploratory Data Analysis (EDA), focuses on _data acquisition and preprocessing_. It involves gathering data from various sources, including APIs and web scraping. The collected data is then subjected to initial cleaning and preprocessing steps to ensure its quality and prepare it for further analysis. This phase sets the foundation for the subsequent EDA and feature engineering stages.

The collection of notebooks that cover this phase is stored in the `notebooks/dp/` directory, where `dp` refers to the two most commonly used expressions to describe this phase, namely "Data Preparation" or "Data Processing".

# Notebooks

## Data Format Conversion: `notebooks/dp/csv_to_parquet.ipynb`

This notebook focuses on converting the dataset from CSV to Parquet format to enhance performance. We assess the efficiency of Parquet and CSV formats in terms of speed, disk space, and data compression. The objective is to identify the optimal format for EDA and feature engineering and carry out the necessary conversion.

# Préambule général

## La modularité à l'ère des notebooks

La doctrine invite à placer l'ensemble de ses *imports* en début de notebook et à avoir une approche séquentielle, consistant à faire dépendre l'état initial d'une cellule à l'état produit par les précédentes.

L'expérience pratique nous a convaincus de ne pas adhérer à cette doctrine.

Dans les notebooks présentés ci-après, l'essentiel des cellules de code sont pensées pour une exécution autonome, sans dépendre de l'exécution préalable de cellules précédentes.

Quand de telles chaînes de dépendances existent, elles sont courtes et localisées à une section élémentaire du notebook.

Cela fait que chaque cellule possède (à quelque exceptions près, cf. ces micro-séquences locales), tous les imports et toutes les intructions de chargement qui permettent de l'exécuter indépendamment de ce qui précède.

Exemple
-------

```Python
from home_credit.load import get_pos_cash_balance
data = get_pos_cash_balance().copy()
data.MONTHS_BALANCE = -data.MONTHS_BALANCE
pivoted = data.pivot(
    index=["SK_ID_CURR", "SK_ID_PREV"],
    columns="MONTHS_BALANCE",
    values=["NAME_CONTRACT_STATUS", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
)
display(pivoted)
```

Un mécanisme de cache fait que si `get_pos_cash_balance` a été appelée dans une cellule précedente, ou que vous exécutez une seconde fois cette cellule, le chargement du fichier n'aura pas lieu une seconde fois, et que votre temps d'attente en sera considérablement réduit.

