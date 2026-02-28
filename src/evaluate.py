

import pandas as pd

import pickle
from sklearn.metrics import accuracy_score

import yaml
import os
import mlflow

# load
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")
params = yaml.safe_load(open(params_path))

train_params = params['train']
mflow_params = params['mlflow']


def evaluate(data_path,model_path):
    data =pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(mflow_params["MLFLOW_TRACKING_URI"])

    model_name = "Best_RandomForestClassifier"
    model_version = "latest"  # you can also use 1 or any version number.

    # Load the model as a PyFuncModel
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("eval_accuracy",accuracy)

    print(f"Model acccuracy :{accuracy}")

    
if __name__=="__main__":
    evaluate(train_params["data"],train_params["model_path"])