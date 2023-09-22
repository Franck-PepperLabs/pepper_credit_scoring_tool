# Fichiers de dashboard Streamlit

Sous-dossiers :
- **`protos/`** : prototypes fonctionnels qui ciblent chacun une fonctionnalité spécifique.
    - **`protos/direct/`** : les prototypes en local sans passage par l'API, donc avec chargement local des modules et des données.
    - **`protos/via_api/`** : les mêmes prototypes (même structure de dossiers et fichiers, mêmes fonctions), mais avec récupération des données via l'API de serving.
- **`prod/`** : l'application intégrée basée sur **`protos/via_api/`** prêt à être déployée sur le serveur d'API.

**Note** Les éléments du dossier **`prod/`** ont été d'abord dans le dossier **`protos/via_api/`** et sont déplacés vers la production lorsqu'ils ont été suffisamment testés et validés. Il n'y résident donc pas en doublon (sauf à démarrer une nouvelle version). En revanche, le fichier éponyme de **`protos/direct/`** est conservé, ce qui fait que **`protos/direct/`** contient davantage de fichiers que **`protos/via_api/`**.

Les version majeures des prototypes sont conservées avec une numérotation de version inverse : le fichier sans suffixe est le fichier actuel, la version précédente est suffixée avec `_1`, la précédente avec `_2` et ainsi de suite, le numéro de version indiquant donc l'ancienneté.

Les fichiers d'application Streamlit utilisent les librairies suivantes :
- ...
- pour les prototypes à accès direct, ... init env ... blabla
- pour les prototypes via API et prod :
    - **`requests`** pour la mise en oeuvre du protocole HTTP avec échange de données au format JSON.

Fichiers :
- **`protos/direct/`**
    - **`customer_data.py`** : ...
    - **`paginate.py`** : ...
    - **`st_ex_1.py`** : ...
- **`protos/via_api/`** : actuellement vide
- **`prod/`** : 
    - **`st_load_table_direct`** : ...
    - **`table_viewer`** : navigateur de tables.
        - Il permet de charger chacune des tables Home Credit à partir d'une liste, les tables brutes de base, comme les tables nettoyées et dérivées. Les colonnes de la table chargée s'affichent sous forme de checkboxes dans la marge de gauche et permettent, par survol, d'obtenir la description de la variable représentée, et de sélectionner ou désélectionner la colonne.

# Lancement et check du serveur d'API

## Lancement et exécution en local

Si la version **`protos/via_api/`** ou **`prod/`**, avoir préalablement lancé le serveur d'API.

Depuis un second terminal, se placer dans le dossier **`{project_dir}/streamlit/`**

```sh
streamlit run ./table_viewer.py
```

Vérifier que tout fonctionne bien :

Se rendre sur http://localhost:8501. L'écran suivant devrait s'afficher :

![streamlit_table_viewer](../../img/serving/streamlit_table_viewer.png)

Traces d'exécution (journalisation) de l'application Streamlit :

![streamlit_server_logging](../../img/serving/streamlit_server_logging.png)


Pour un lancement de Streamlit directement depuis l'URL GitHub de l'application :

```sh
streamlit run https://raw.githubusercontent.com/Franck-PepperLabs/pepper_credit_scoring_tool/main/streamlit/prod/table_viewer.py
```

