

import pandas as pd

import pickle
from sklearn.metrics import accuracy_score

import yaml
import os
import mlflow


# load
train_params = yaml.safe_load(open("params.yaml"))['train']
mlflow_params = yaml.safe_load(open("params.yaml"))['mlflow']


def evaluate(data_path,model_path):
    data =pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

    # Load the model as a PyFuncModel
    model_name = "Best_RandomForestClassifier"
    model_version = 1 # 'latest'
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("eval_accuracy",accuracy)

    print(f"Model acccuracy :{accuracy}")

if __name__=="__main__":
    evaluate(train_params["data"],train_params["model_path"])