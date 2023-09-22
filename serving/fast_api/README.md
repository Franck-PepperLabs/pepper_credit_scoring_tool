# Modules utilisés par l'API

- **`home_credit.api.get_table_names`** : retourne la liste des tables disponibles.
- **`home_credit.api.get_table_range`** : retourne une plage de lignes de l'une des tables disponibles.

# Fichiers du serving d'API

- **`fast_api_main.py`** : application de serving principale. Elle produit la route `/` ("Welcome") et délègue aux routeurs secondaires la gestion des autres routes, organisées par modules spécialisés.
    - **utilise** :
        - `fastapi.FastAPI` comme framework d'application de serving.
        - `logging` pour la gestion de la journalisation
        - `os`, `sys`, `dotenv` pour l'initialisation de l'environnement à partir d'un fichier `.env`
        - les routeurs de `get_table_names` et `get_table`.
    - **utilisée par** : lancement du serveur d'API
- **`get_table_names.py`** (route `/api/table_names`) : retourne la liste des tables disponibles.
    - **utilise** : `home_credit.api.get_table_names`.
    - **utilisée par** :
        - `st_load_table_direct`, `st_load_table_api_v2` pour alimenter la liste de sélection de la table à afficher.
        - `customer_data.py` pour itérer sur l'ensemble des tables et afficher les informations qu'elle contient sur un client donné. NB > actuellement, ce n'est qu'un prototype avec sa propre fonction `get_table_names` câblée en dur.
- **`get_table.py`** (route `/api/table`) : retourne une plage de lignes de l'une des tables disponibles.
    - **utilise** : `home_credit.api.get_table_range`.

# Lancement et check du serveur d'API

## Lancement et exécution en local

Depuis un terminal, se placer dans le dossier **`{project_dir}/serving/`** et lancer le serveur :

```sh
uvicorn fast_api_main:app --reload
```

Vérifier que tout fonctionne bien :

Se rendre sur http://localhost:8000. L'écran suivant devrait s'afficher :

![fast_api_welcome](../../img/serving/fast_api_welcome.png)

Se rendre sur http://localhost:8000/docs. L'écran de la documentation d'API Swagger devrait s'afficher :

![fast_api_docs_swagger](../../img/serving/fast_api_docs_swagger.png)

Se rendre sur http://localhost:8000/redoc. L'écran de la documentation d'API Redocly devrait s'afficher :

![fast_api_docs_redocly](../../img/serving/fast_api_docs_redocly.png)

Traces d'exécution (journalisation) du serveur d'API :

![fast_api_server_logging](../../img/serving/fast_api_server_logging.png)

# Déploiement et exécution sur une offre Oracle Cloud Free Tier

Services **ALWAYS FREE** : de vraies ressources, toujours gratuit, jamais d'interruption.

## Création d'un compte

Créer un compte, activer la MFA, installer l'application Oracle Mobile Authenticator.

Pour la suite, s'assurer de consommer uniquement des services estampillés **ALWAYS FREE** ou **TOUJOURS GRATUIT**.

## [**Création d'une instance de calcul (une VM)**](https://cloud.oracle.com/compute/instances/create?region=eu-marseille-1)

Créez une instance pour déployer et exécuter des applications ou enregistrez la ressource en tant que pile Terraform réutilisable pour créer une instance à l'aide de Resource Manager.

### Nom et emplacement

**Nom** : proposition par défaut `instance-{instance-id}` changée pour `home-credit-fast-api`

**Créer dans le compartiment** : `{main-user} (racine)`

### Placement
- **Domaine de disponibilité** : `AD-1` (_Admissible à Toujours gratuit_)
- **Domaine de pannes** : Laisser Oracle choisir le meilleur domaine de pannes
- **Type de capacité** : Capacité à la demande

### Sécurité

- **Instance protégée** : Désactivé

### Image et forme

- **Image** : `Oracle Linux 8`
- **Build d'image** : `2023.08.31-0`
- **Forme** : `VM.Standard.E2.1.Micro` (_Admissible à Toujours gratuit_)
- **Nombre d'OCPU** : 1
- **Mémoire (Go)** :  1
- **Bande passante réseau (Gbits/s)** : 0.48

### Informations sur la carte d'interface réseau virtuelles principale

- **Nouveau nom du réseau cloud virtuel** : `vcn-{instance_id}`
- **Créer dans le compartiment** : `{main-user} (racine)`
- **Nouveau nom du sous-réseau** : `subnet-{instance-id}`
- **Créer dans le compartiment** : `{main-user}(racine)`
- **Options de lancement** : `-`
- **Bloc CIDR** : `10.0.0.0/24`
- **Affecter une adresse IPv4 publique** : Oui
- **Enregistrement DNS** : Oui

Carte d'interface réseau virtuelle secondaire _Facultatif_

En plus d'une carte d'interface réseau virtuelle principale, vous pouvez connecter une carte d'interface réseau virtuelle secondaire à votre instance. Cette opération peut être réalisée lors du lancement de l'instance ou après.

Aucune carte d'interface réseau virtuelle à afficher

### Ajouter des clés SSH

Générez une paire de clés SSH pour vous connecter à l'instance à l'aide d'une connexion SSH (Secure Shell) ou téléchargez une clé publique que vous possédez déjà.

- ✔ Générer une paire de clés pour moi
- Télécharger les fichiers de clés publiques (.pub)
- Coller des clés publiques
- Aucune clé SSH

Téléchargez la clé privée pour pouvoir vous connecter à l'instance via SSH. Elle ne sera plus affichée.

- Enregistrer la clé privée -> **`{project-dir}/_ssh/ssh-key-{today}.key`**
- Enregistrer la clé publique

### Volume d'initialisation

Un volume d'initialisation est un appareil amovible qui contient l'image servant à démarrer l'instance de calcul.

- **Indiquer une taille personnalisée de volume d'initialisation** - Les performances d'un volume varient en fonction de sa taille. Taille de volume d'initialisation par défaut : 46.6 Go. Si vous indiquez une taille de volume d'initialisation personnalisée, les limites de service sont appliquées.
- ✔ **Utiliser le cryptage en transit** - Crypte les données en transit entre l'instance, le volume d'initialisation et les volumes de blocs.
- **Crypter ce volume avec une clé gérée par vous-même** - Par défaut, Oracle gère les clés qui cryptent ce volume, mais vous pouvez choisir une clé dans un coffre auquel vous avez accès si vous souhaitez plus de contrôle sur le cycle de vie et l'utilisation de la clé. Comment gérer mes propres clés de cryptage ?

### Options avancées

On passe

### Création de l'instance

Bouton Créer..

Adresse IP publique: `{instance-ip-addr}`

Nom de domaine: 

![oracle_home_credit_fast_api_instance](../../img/serving/oracle_home_credit_fast_api_instance.png)

### Mise à jour de l'environnement

#### Connexion à l'instance

```sh
chmod 400 {project_dir}/_ssh/ssh-key-{today}.key
```
ou même 

```sh
chmod -R 400 {project_dir}/_ssh
```

```sh
ssh -i {project_dir}/_ssh/ssh-key-{today}.key opc@{instance-ip-addr}
```

```sh
[opc@home-credit-fast-api ~]$ pwd
/home/opc
```
#### Mise à jour des paquets disponibles

```sh
sudo yum update
```

Très, très, très, très long.

#### Installation de Python

Check de la version pré-installée :

```sh
[opc@home-credit-fast-api ~]$ python --version
Python 3.6.8
```

```sh
sudo yum install python311
```

Check

```sh
[opc@home-credit-fast-api ~]$ python3.11 --version
Python 3.11.2
```

### Déploiement et test d'accessibilité

#### Copie des fichiers de serving Fast API

```sh
scp -i {project-dir}/_ssh/ssh-key-{today}.key -r {project-dir}/serving/fast_api/ opc@{instance-ip-addr}:/home/opc/
```

```sh
[opc@home-credit-fast-api ~]$ ls fast_api/
fast_api_main.py  get_table_names.py  get_table.py
```

#### Production de `requirements.txt`

```sh
pip show pipreqs
```

```sh
python -m pip install pipreqs
```

```sh
cd {project-dir}/serving/fast_api/
pipreqs ./ --encoding=utf-8
cd {project-dir}/python/
pipreqs ./ --encoding=utf-8

```

#### Installation des dépendances Python

On refait (avec les deux `requirements.txt` précédents fusionnés)

```sh
scp -i {project_dir}/_ssh/ssh-key-{today}.key -r {project-dir}/serving/fast_api/ opc@{instance-ip-addr}:/home/opc/
```

Côté serveur :

```sh
[opc@home-credit-fast-api ~]$ cd fast_api/
[opc@home-credit-fast-api fast_api]$ ls
fast_api_main.py  get_table_names.py  get_table.py  requirements.txt
[opc@home-credit-fast-api fast_api]$ python3.11 -m pip install -r requirements.txt
/usr/bin/python3.11: No module named pip
```

Installation de pip :

```sh
[opc@home-credit-fast-api fast_api]$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
[opc@home-credit-fast-api ~]$ python3.11 get-pip.py
```

Installation des packages Python requis (note : il y en a plus qu'il n'en faut dans cette configuration)

```sh
[opc@home-credit-fast-api fast_api]$ python3.11 -m pip install -r requirements.txt
```

#### Copie du code Python

Les deux packages `pepper` et `home_credit` sont à copier dans le dossier `fast_api`. Cela permet à FastAPI d'y accéder. L'idéal serait d'utiliser un gestionnaire de paquets, mais pas le temps sur cette early version de démonstration.

```sh
scp -i {project_dir}/_ssh/ssh-key-{today}.key -r {project-dir}/python/pepper/ opc@{instance-ip-addr}:/home/opc/fast_api/
```

Oui, mais pas le dossier `__pycache__` ni son contenu !

Donc (pas d'option `--exclude` avec `scp`) :

```sh
[opc@home-credit-fast-api fast_api]$ cd pepper
[opc@home-credit-fast-api pepper]$ rm -r __pycache__/
```

```sh
scp -i {project_dir}/_ssh/ssh-key-{today}.key -r --exclude=__pycache__ {project-dir}/python/home_credit/ opc@{instance-ip-addr}:/home/opc/fast_api/
```

```sh
[opc@home-credit-fast-api fast_api]$ cd ../home_credit
[opc@home-credit-fast-api pepper]$ rm -r __pycache__/
```

#### Lancement du serveur

À ajouter à `requirements.txt`, il ne vient pas avec la distribution Fast API.
```sh
python3.11 -m pip install uvicorn
```

```sh
uvicorn fast_api_main:app --host 0.0.0.0 --port 8000 --reload
```

L'adresse IP `0.0.0.0` signifie qu'il écoutera toutes les interfaces réseau et sera accessible depuis l'extérieur. 

L'option `--reload` permettra au serveur de se recharger automatiquement lorsque les fichiers sont mis à jour.

Erreur de locale au premier lancement : "fr_FR.UTF-8" non prise en charge.. 

```sh
[opc@home-credit-fast-api fast_api]$ locale -a
```

Il n'y a que du `en_..`

```sh
sudo yum install glibc-langpack-fr
export LC_ALL=fr_FR.UTF-8
```

Check `http://{instance-ip-addr}:8000`.

Ca ne répond pas..

Depuis le serveur, connecté en SSH :

```sh
[opc@home-credit-fast-api ~]$ curl http://127.0.0.1:8000/
{"message":"Welcome to Home Credit Dashboard"}[opc@home-credit-fast-api ~]$
```

Donc le serveur répond. C'est un problème de firewall.

### Compléter la configuration réseau et sécurité

Il s'agit d'ouvrir la route qui mène de l'Internet au port sur lequel écoute et répond l'application de service d'API.



Retour sur la page de l'instance :

Carte d'interface réseau virtuelle principale :
- Adresse IPv4 publique: 129.151.252.170
- Adresse IPv4 privée: 10.0.0.41
- Groupes de sécurité réseau: Aucun [**Modifier**]
- Sous-réseau:subnet-20230922-1259
- Enregistrement DNS privé: Activer
- Nom d'hôte: home-credit-fast-api
- Nom de domaine qualifié complet interne: home-credit-fast-api...
- home-credit-fast-api.subnet09221326.vcn09221326.oraclevcn.com

Modifier : ça ne fonctionne pas, car il n'y a pas de groupe de sécurité défini

Accès direct à : Default Security List for vcn-20230922-1259 (liste de sécurité par défaut du sous-réseau virtuel associé)

3 règles entrantes actuellement :

|Sans conservation de statut|Source|Protocole IP|Plage de ports source|Plage de ports de destination|Type et code|Autorise|Description|
|-|-|-|-|-|-|-|-|
|Non|0.0.0.0/0|TCP|Tout|22||Trafic TCP pour les ports : 22 SSH Remote Login Protocol||
|Non|0.0.0.0/0|ICMP|||3, 4|Trafic ICMP pour : 3, 4 Destination inaccessible: La fragmentation est requise mais l'option Ne pas fragmenter a été définie||
|Non|10.0.0.0/16|ICMP|||3|Trafic ICMP pour : 3 Destination inaccessible||
|Nouvelle règle|0.0.0.0/0|TCP|Tout|8000||Trafic TCP pour le port 8000 FastAPI||


**Rien à faire !!**

Recherche Stack Overflow : https://stackoverflow.com/questions/62326988/cant-access-oracle-cloud-always-free-compute-http-port


Le problème est ressemblant, mais pas identique : dans l'approche préconisé, il y a le paramétrage que nous avons effectué ci-dessus, puis un paramétrage du firewall du système, en mode restreint, ce qui est la bonne pratique de sécurité.


Mais pour ce qui nous concerne, avant de réduire l'accès, il faudrait déjà l'ouvrir.

Vérification rapide :

```sh
sudo iptables -L INPUT  # la table INPUT qui peut être responsable du blocage
[opc@home-credit-fast-api ~]$ sudo iptables -L INPUT
Chain INPUT (policy ACCEPT)
target     prot opt source               destination
```

La table est vide, il n'y aucune restriction d'accès du trafic entrant.

Autre vérification :

```sh
[opc@home-credit-fast-api ~]$ netstat -tuln | grep 8000
tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN
```

C'est bien à l'écoute sur le port 8000, donc le serveur est ouvert et à l'écoute du monde extérieur.

Le problème est donc (a priori, sous réserve de bon diagnostic) du côté du paramétrage du VPN Oracle.

Sauf erreur, il n'est explicitement besoin de créer un groupe de sécurité, mais c'est peut-être là qu'est le problème.

