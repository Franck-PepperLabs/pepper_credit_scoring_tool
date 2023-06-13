# Data Project Steps

This project follows a structured approach, organized into different phases within the `notebooks/` directory. Each phase represents a specific stage in the data analysis and modeling process.

By following this systematic workflow, the project progresses through each phase, covering essential steps from data acquisition to advanced modeling and evaluation. Each notebook within the respective phase folder provides a clear progression in the project's data-driven journey.

By following this systematic workflow, the project progresses through each stage, covering essential steps from data acquisition to advanced modeling and evaluation. These stages are not strictly sequential, and there can be iterations and back-and-forth between them, following an iterative and incremental process typical of agile methodologies.

## Stage 1: Data Acquisition (DA)

The Data Acquisition stage involves gathering data from various sources, including databases, files, APIs, web scraping, as well as data extraction techniques used in CV, NLP, audio, or unstructured data (corpora). This stage encompasses both Data Extraction, where data is generated or extracted from unstructured sources, and Data Gathering (Data Collection), where data is collected from existing sources.

## Stage 2: Data Preprocessing (DP)

After the data is acquired, it undergoes preprocessing tasks to ensure its quality, consistency, and usability. These tasks include data cleaning, handling missing values, data format standardization, normalization, and initial feature engineering. The objective is to prepare the data for the subsequent Exploratory Data Analysis (EDA) and more advanced feature engineering stages.

## Stage 3: Exploratory Data Analysis (EDA)

In this stage, the collected and preprocessed data is thoroughly explored to gain a deeper understanding of its structure, relationships, and patterns. Statistical measures, visualizations, and correlation analysis are used to identify meaningful insights, important variables, and potential issues. The EDA stage helps guide the subsequent feature engineering and modeling decisions.

## Stage 4: Feature Engineering (FE)

Once the data is prepared and explored, the Feature Engineering stage begins. This stage focuses on transforming the existing variables and creating new ones to enhance the predictive power of the data. Techniques such as encoding categorical variables, scaling numerical features, handling outliers, creating interaction terms, and deriving new informative features are applied. Feature engineering aims to capture the underlying patterns and relationships in the data that can improve the performance of the models.

## Stage 5: Model Development and Evaluation (ML)

This stage focuses on developing models based on the engineered features and evaluating their performance. It can be further divided into the following sub-stages:

### Sub-stage 1: Model Benchmarking and Selection

In this sub-stage, a benchmark of candidate models is prepared. Models suitable for the problem at hand are selected based on the project objectives and insights gained from EDA. This includes choosing a reference dummy model, employing standard discovery models, and establishing performance benchmarks, such as using random forests. Selection of performance metrics is also tailored to the project's nature (e.g., R2 for regression, ARI for clustering, ROC AUC for binary classification, etc.).

### Sub-stage 2: Model-Specific Preprocessing

Each candidate model may require specific preprocessing steps to ensure compatibility with the training data. This sub-stage involves tasks such as dimensionality reduction, handling class imbalance, and adapting the data to suit the requirements of each model.

### Sub-stage 3: Hyperparameter Tuning

Grid search techniques are employed to explore the hyperparameter space of the candidate models. This process involves working with sample datasets and utilizing cross-validation techniques to avoid overfitting. Attention is given to selecting the optimal hyperparameters that yield the best model performance.

### Sub-stage 4: Model Optimization

Once the best hyperparameters are identified, this sub-phase focuses on further optimizing the selected model. Advanced optimization techniques are applied, which may include parameter tuning methods specific to the chosen model. Additionally, preliminary evaluation of data drift can be performed using only the training data.

### Sub-stage 5: Model Interpretation and Explainability

Understanding how the models work and interpreting their predictions is of utmost importance. Techniques such as feature importance analysis, partial dependence plots, SHAP values, and model-specific interpretability methods are applied to gain insights into the model's decision-making process and provide explanations for its predictions.

## Stage 6: Model Deployment and Monitoring (MS)

Once the final models are selected, the deployment stage begins. The models are saved and prepared for production deployment. Infrastructure is set up to support the deployed models, including API integration, batch processing, or real-time serving. Monitoring systems are established to track the models' performance, detect data drift, and ensure the models remain effective and up-to-date in a real-world environment.

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

