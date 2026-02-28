# mlops_aws
Cette repository contient la mise en place d'une simple (formulaire web) basé sur un model IA exploité via un API Flask.
L'objective c'est d'avoir ce modèle suivant un pipeline MLOPS complet, déployé sur l'infrastructure AWS.
Les technologies utilisé : 
Python, Flask,
MLFLow, dvc,
AWS EC2, AWS IAM, AWS S3,
HTML, CSS, JS (maybe)

# 1. Mise de l'environement MLFLOW sur AWS : 

Le but de cet etape et des crée un serveur MLFlow, qui gère le registry de nos experiences MLs:

1. Allez dans votre compte AWS,
2. Crée un bucket s3, (rendre son accès public - pour minimiser les configuration nécessaire entre s3 et MLFLOW)
4. Crée un clé d'accès (Access Key) et  telecharger la, dans le fichier csv (vous trouvez votre Access_key_Id et Votre Secret_key).
5. Crée une machine Ec2 (type medium, OS:ubuntu), crée un key paire pour cette machine. donnez lui un accès Custom Port 5000 avec source 0.0.0.0/0 (tous le monde), 
6. sur cette machine on doit configurer notre MLFLOW avec les commandes suivantes (y inclut l'installation de quelques dependances) : 
```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
mkdir mlflow
cd mlflow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell
```
6. Maintenant configurer les accès à s3 pour MLflow depuis la machine (grace au clé que vous avez crée pour l'utilisateur dans l'etape 4)

```bash
# Set aws credentials
aws configure
```

7. Maintenant vous pouvez lancer mlflow dans cette machine en y précisant le nom de votre bucket (NB: le bucket sera utilisé pour stoker le cashe de MLFLOW : les runs, les experiments, les artefacts loggué, les models ...)

```bash
    # run mlflow server to be accessible globaly
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://YOUR_BUCKET_NAME --allowed-hosts "*" --cors-allowed-origins "*" 
```

6. [Conditionel] Si vous avez skipper le custom port 5000 accès vous pouvez allez dans la liste des instances EC2, clicker sur l'ID de votre instance, et dans l'onglet Security clicker sur le nom de votre security group,
et ajouter une nouvelle règle dans les inboud rules. 
_Custom - Port 5000 - 0.0.0.0/0_

8. Mlflow est running donc vous pouvez l'accéder depuis l'adresse le lien `http://ADDRESS_IP_EC2:5000`

9. Si vous avez stopper la machine et vous avez connecter à nouveau il faut reactiver l'environement pipenv et relancer votre mlflow.
> pipenv est un gestionaire d'environement python basé sur le dossier (dans ce cas "mlflow") ce n'est pas commes les autres packageurs d'environnement qui peuvent etre activé globalement avec un chemin absolue.
```bash
    cd mlflow
    pipenv shell
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://YOUR_BUCKET_NAME --allowed-hosts "*" --cors-allowed-origins "*" 
```

10. Maintenant vous pouvez exploiter MLFlow pour faire le suivi de vos modèles (en précisant le tracking uri dans votre code ) :

```python
import mlflow
mlflow.set_tracking_uri("http://ADDRESS_IP_EC2:5000")
```

> TRES BIEN VOUS AVEZ COMPLETE CETTE ETAPE.

# 2.Repository git et Code de départ :

Le but de cet etape est d'avoir un code de départ qui simule quelques etapes classique dans le processus ML (preprocess, train, evaluate) (des étapes qui sont utilisé dans le 'Building' de modèles.)

> CI Traditionel : Code Change --> Build --> Unit Tests --> Artifact
> ML CI : Code Change OU **Data Change** --> Envirnoment Setup --> Dvc Repro (lance toutes les phases) --> MLFlow Logging (Registry)

> Dans le DevOps traditionel uniquement le code change (git) peut changer le système (l'artéfact) finale, alors que dans le MLOps on 'a pas uniquement le code change mais aussi les data change (dvc).

> Ceci implique que nous avons un outil pour le versionning code --> Git

> Et un outil de versionning de données --> dvc

11. Crée un dossier de travail et cloner cette repository dedans
  `git clone : git@github.com:yehdihouldanna/mlops_aws.git`

12. Allez dans votre compte github et crée une nouvelle repository (laisser la vierge) (`github/username/YOUR_REPOSITORY_NAME`)

13. Dans le dossier du repository cloné (etape 11) : Initer git et mettez le remote à votre repository, et pusher le code cloné sur votre repository

```bash 
git init
git remote add origin git@github.com:USERNAME/YOUR_REPOSITORY_NAME.git
git add .
git commit -m "first commit : Starting Code"
git push
```

# 3. Data Originale (Source de données) :
Pour similer une situation du monde réel avec une source de données externe dynamique nous allons crée sur notre bucket s3 un dossier dédié pour les données raw (ceci pourra venir de plusieurs sources).

Dans votre bucket S3 crée les dossier ``data/raw`` et uploader votre fichier ``diabetes.csv``
en meme temps crée aussi le dossier ``data/processed``

Donc vous devrez avoir ce chemin dans votre bucket s3
`s3://YOUR_BUCKET_NAME/data/raw/diabetes.csv`

> Dans le monde réel ceci pourra simuler le resultat d'un ETL lourd

# 3.Repository dvc sur s3 :

> Il existe des outils pour faire le repository dvc, cependant il faut comprendre que le dvc crée un fichier d'historique des metadata des données (elle ne stocke pas les données, mais chaque push de dvc correspond à un etat 'metadata' de données, elle peut contenir le path, sans contenir les données eux memes.)
> Cependant nous allons heberger notre repository dvc sur s3 c'est simple comme ça.
> Du coup on a deux chose à garder sur notre s3, nos données de base (n'a rien avoir avec dvc nécessairement, on peut avoir des données qui sont misees à jour journalierement)

14. initier dvc avec `dvc init`
15. crée une repository dvc  (attention : il faut crée cette dossier dans le meme dossier que votre git.)
(La ceci est le store c'est à dire il contient notre tree dvc)

```bash
    dvc init
    conda install dvc[s3]

    # 1. Define where DVC should store its actual "packages" (The Cache)
    dvc remote add -d dvcstore s3://YOUR_BUCKET_NAME/dvc_repo

    # 2. Tell DVC to use your AWS credentials
    dvc remote modify dvcstore access_key_id   VOTRE_ACCESS_KEY_ID
    dvc remote modify dvcstore secret_access_key VOTRE_SECRET_ACCESS_KEY
```
16. Ajouter la source de données à tracker : 

```bash
    dvc import-url s3://YOUR_BUCKET_NAME/data/ .
    #dvc import-url --to-remote s3://YOUR_BUCKET_NAME/data/ .
    dvc push # now you can check on your s3 to see the result
```
> --to-remote permet de n'a pas avoir une copie local du fichier dans le dossier du code (la copie sera telechargé ailleur et son hash md5 sera claculé et sera supprimé après), si vous l'enlever le fichier sera dans votre dossier data

# 4. Adjuster votre fichier params :

ex:
```yaml
preprocess:
  #input : data/raw/diabetes.csv
  #output : data/preprocessed/data.csv
  input : data/raw/diabetes.csv # for this to work you need to pull from dvc first or bring the file manually.
  output : s3://VOTRE_BUCKET_S3/data/processed/data.csv

train : 
  data : s3://VOTRE_BUCKET_S3/data/processed/data.csv
  model_path : models.pkl
  random_state : 42
  n_estimators: 100
  max_depth: 5

mlflow: 
  MLFLOW_TRACKING_URI : http://EC2-PUBLIC-IP-ADRESS:5000
aws :
  aws_access_key_id: YOUR_ACCESS_KEY
  aws_secret_access_key: YOUR_SECRET_ACESS_KEY
  region_name: AWS_REGION
```
> Ce fichier assume que votre code preprocessing (utilise des données diabetes csv local [ou traked avec dvc et pulled]), applique le preprocessing pour sauvergarder une version preprocessed sur s3, et utlise la version sauvegardé sur s3 pour faire le training.
> EN suite le fichier evaluate utilise le mlflow running sur ec2 pour recuperer la version nommé dans le code de votre modèle pour l'appeler.


# La commande qui automatise le lancement des fichiers
Maintenant avec la façon dont notre code est structuré en phases (preprocess et train)
nous pouvons lancer tous le flow quand on veut avec 

```bash
dvc repro
```
