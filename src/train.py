import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle 
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split,GridSearchCV


# dagshub experiment tracking uri : found in experiment onglet
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/yehdihouldanna/mlpipeline-mlflow-dvc.mlflow"

os.environ["MLFLOW_TRACKING_USERNAME"] = 'yehdihouldanna' #dagsub username

# this is the token in your profile (default) and you can also copy it from the dvc onglet in the repository
os.environ["MLFLOW_TRACKING_PASSWORD"]= "28fdecc4cae73c08dadfd562c7bfe688f20b9a78"



import mlflow
from mlflow.models import infer_signature
import mlflow.pyfunc

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=2,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

train_params = yaml.safe_load(open("params.yaml"))['train']
dagshub_params = yaml.safe_load(open("params.yaml"))['dagshub']

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(dagshub_params["MLFLOW_TRACKING_URI"])
    with mlflow.start_run():
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

        signature = infer_signature(X_train,y_train)

        # Define hyper params grid: 
        
        param_grid = {
            # 'n_estimators' : [100, 200],
            # 'max_depth' : [5,10,None],
            # 'min_samples_split' : [2,5],
            'min_samples_leaf' : [1,2]
        }

        # Perform hte hyperparams tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        # get the best model : 
        best_model = grid_search.best_estimator_

        # predict and evaluate the model

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_params(grid_search.best_params_)

        # log the confusions matrix and  classification report : 
        cm = confusion_matrix(y_test,y_pred)
        cr = classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(str(cr),"classification_report.txt")

        # # To solve some errors related to wrong file type while logging the model
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # if tracking_url_type_store !="file":
        #     mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best_model")
        # else : 
        # mlflow.sklearn.log_model(
        #     sk_model=best_model,
        #     artifact_path="model",
        #     signature=signature
        # )
        
        # Replace the mlflow.sklearn.log_model() call with:
        
        #? mlflow.sklearn.save_model(
            #? sk_model=best_model,
            #? path="model_path",
        #? )
        #? # Then log the model as an artifact
        #? mlflow.log_artifacts("model_path")

        import joblib
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

        mlflow.log_artifact(model_path)


        # creaete the directory to save the model : 
        # os.makedirs(os.path.dirname(model_path),exist_ok=True)

        # filename=model_path
        # pickle.dump(best_model,open(filename,"wb"))
        # print(f"Model saved to {model_path}")


if __name__=="__main__":
    train(train_params["data"],train_params["model_path"],train_params["random_state"], train_params["n_estimators"],train_params['max_depth'])







