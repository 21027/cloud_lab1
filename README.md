# mlops_aws
Cette repository contient la mise en place d'une simple (formulaire web) basé sur un model IA exploité via un API Flask.
L'objective c'est d'avoir ce modèle suivant un pipeline MLOPS complet, déployé sur l'infrastructure AWS.
Les technologies utilisé : 
Python, Flask,
MLFLow, dvc,
AWS EC2, AWS IAM, AWS S3,
HTML, CSS, JS (maybe)

# 1. Mise de l'environement MLFLOW sur AWS : 
1. Allez dans votre compte AWS,
3. Crée un bucket s3, (rendre son accès public - pour minimiser les configuration nécessaire entre s3 et MLFLOW)
4. Crée un utilisateur dans IAM,  et crée un clé d'accès et le telecharger.
5. Crée une machine Ec2 (type medium, OS:ubuntu), crée un key paire pour cette machine. donnez lui un accès Custom Port 5000 avec source 0.0.0.0/0 (tous le monde), 
5.5 Si vous avez skipper le custom port 5000 accès vous pouvez allez dans la liste des instances, clicker sur l'ID de votre instance, et dans l'onglet Security clicker sur le nom de votre security group, (image 1)
et ajouter la nouvelle règle dans les inboud rules.
6. sur cette machine on doit configurer notre MLFLOW avec les commandes suivantes : 
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
6. Maintenant configurer les accès à s3 pour MLflow depuis la machine (grace au clé que vous avez crée pour l'utilisateur)
```bash
# Set aws credentials
aws configure
```

7. Maintenant vous pouvez lancer mlflow dans cette machine en y précisant le nom de votre bucket (NB: le bucket sera utiliser pour stoker le cash de MLFLOW les runs, les experiments, les artefacts logé, les models ...)
```bash
    # run mlflow server to be accessible globaly
    mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root --allowed-hosts "*" --cors-allowed-origins "*" s3://YOUR_BUCKET_NAME
```

8. Mlflow est running donc vous pouvez l'accéder depuis l'adresse le lien `http://ADDRESS_IP_EC2:5000`
9. Si vous avez stopper la machine et vous avez connecter à nouveau il faut reactiver l'environement pipenv et relancer votre mlflow.
> pipenv est un gestionaire d'environement python basé sur le dossier (dans ce cas "mlflow") ce n'est pas commes les autres packageurs d'environnement qui peuvent etre activé globalement avec un chemin absolue.
```bash
cd mlflow
pipenv shell
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root --allowed-hosts "*" --cors-allowed-origins "*" s3://YOUR_BUCKET_NAME
```

10. Maintenant vous pouvez ajouter ce lien `http://ADDRESS_IP_EC2:5000` comme votre mlflow_tracking_uri dans vos scripts ou notebook locales,
et grace à ça les modèles que vous crée sont maintenant en ligne.


# 2.Repository git:
11. Cloner cette repository git localement dans votre machine
12. Ajouter les fichier que vous avez crée dans votre code dans le dossier contenant le git
13. Faitez un push : `git add .`; `git commit -m "first commit"`; `git push`

# 3.Repository dvc sur s3 :
14. initier dvc avec `dvc init`
15. crée une repository dvc 

```bash
    dvc init
    conda install dvc[s3]
    dvc remote add -d dvcstore s3://YOUR_BUCKET_NAME/dvc
    dvc remote modify dvcstore access_key_id   *******
    dvc remote modify dvcstore secret_access_key ********
    dvc add data/raw
    dvc push # now you can check on your s3 to see the result
```

# 4. Adjuster votre fichier params :

ex:
```yaml
preprocess:
  #input : data/raw/diabetes.csv
  #output : data/preprocessed/data.csv
  input : data/raw/diabetes.csv # for this to work you need to pull from dvc first or bring the file manually.
  output : s3://VOTRE_BUCKET_S3/data/preprocessed/data.csv

train : 
  data : s3://VOTRE_BUCKET_S3/data/preprocessed/data.csv
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

## Complemenatires  :
1. Comment connecter à votre machine EC2 depuis votre propre terminale:
```bash
ssh -i path/to/your-key.pem ec2-user@<your-ec2-public-ip>
# ex:
ssh -i "C:\Users\yana\Downloads\mlflow-server-ya.pem" ubuntu@13.53.36.3
```





image 1 :
![alt text](image.png)