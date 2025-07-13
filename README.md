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
2. Crée un utilisateur dans IAM, avec permission admin, et crée un clé d'accès et le telecharger.
3. Crée un bucket s3, (rendre son accès public - pour minimiser les configuration nécessaire entre s3 et MLFLOW)

4. Crée une machine Ec2 (type medium, OS:ubuntu), crée un key paire pour cette machine.
5. sur cette machine on doit configurer notre MLFLOW avec les commandes suivantes : 
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
mlflow server -h 0.0.0.0 --default-artifact-root s3://YOUR_BUCKET_NAME
```

8. Mlflow est running donc vous pouvez l'accéder depuis l'adresse le lien `http://ADDRESS_IP_EC2:5000`
9. Si vous avez stopper la machine et vous avez connecter à nouveau il faut reactiver l'environement pipenv et relancer votre mlflow.
> pipenv est un gestionaire d'environement python basé sur le dossier (dans ce cas "mlflow") ce n'est pas commes les autres packageurs d'environnement qui peuvent etre activé globalement avec un chemin absolue.
```bash
cd mlflow
pipenv shell
mlflow server -h 0.0.0.0 --default-artifact-root s3://YOUR_BUCKET_NAME
```

10. Maintenant vous pouvez ajouter ce lien `http://ADDRESS_IP_EC2:5000` comme votre mlflow_tracking_uri dans vos scripts ou notebook locales,
et grace à ça les modèles que vous crée sont maintenant en ligne.



> Une façon simple pour accéder à votre machine EC2 depuis votre terminale : 
```bash
ssh -i path/to/your-key.pem ec2-user@<your-ec2-public-ip>
# ex:
ssh -i "C:\Users\yana\Downloads\mlflow-server-ya.pem" ubuntu@13.53.36.3
```

