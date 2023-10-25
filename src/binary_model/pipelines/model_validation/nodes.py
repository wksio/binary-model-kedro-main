"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.18.9
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import datasets
import pandas as pd
from typing import Tuple, Dict
import os
from sklearn.ensemble import BaggingClassifier
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def cross_validation_sfs(sfs_model:object, data:pd.DataFrame, y:str, tuned_logistic_params:Dict, cv_params:Dict) -> None:
    X = sfs_model.transform(data)
    Y= data[y]
    if "max_iter" in tuned_logistic_params:
        max_iter=tuned_logistic_params["max_iter"]
    else:
        max_iter=10000
    logistic_model = LogisticRegression(
        solver=tuned_logistic_params["solver"],
        max_iter=max_iter,
        C=tuned_logistic_params["C"],
        penalty=tuned_logistic_params["penalty"]
    )
    sk_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    scores = cross_val_score(logistic_model, X, Y, cv = sk_folds)

    with open(os.path.join(cv_params["report_path"], cv_params["report_name"]), "w") as f:
        f.write("Cross Validation Scores: " + str(scores))
        f.write("\n")
        f.write("Average CV Score: " + str(scores.mean()))
        f.write("\n")
        f.write("Number of CV Scores used in Average: " +str(len(scores)))
        f.write("\n")
        f.write("Standard Deviation of CV Scores: " +  str(scores.std()))

def hyperparameter_tunning(sfs_model:object, X_train:pd.DataFrame, y_train:pd.DataFrame,hyper_params:Dict,split_params:Dict) -> object:
    # Create logistic regression model
    X = sfs_model.transform(X_train)
    Y= y_train
    logistic_model = LogisticRegression(
        max_iter=10000,
    )
    hyper_params['C'] = loguniform(1e-5, 100)
    cv = RepeatedStratifiedKFold(n_splits=split_params["n_splits"], n_repeats=split_params["n_repeats"], random_state=split_params["random_state"])
    
    search = RandomizedSearchCV(logistic_model, hyper_params, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

    result = search.fit(X, Y)
    return result

def generate_tunning_report(result:object,tunning_report:Dict) -> None:
    with open(os.path.join(tunning_report["report_path"], tunning_report["report_name"]), "w") as f:
        f.write("Best Score: " + str(result.best_score_))
        f.write("\n")
        f.write("Best Hyperparameters: " + str(result.best_params_))
        
def extract_best_params(result:object) -> Dict:
    return result.best_params_

def comparison_model(forward: object, forward_sfs_model: object, backward: object, backward_sfs_model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, final_params:Dict) -> Tuple:
    forward_x=forward_sfs_model.transform(X_train)
    forward_x_test=forward_sfs_model.transform(X_test)
    forward_model = LogisticRegression(
        solver=forward.best_params_["solver"],
        max_iter=10000,
        C=forward.best_params_["C"],
        penalty=forward.best_params_["penalty"]
    )
    backward_x=backward_sfs_model.transform(X_train)
    backward_x_test=backward_sfs_model.transform(X_test)
    backward_model = LogisticRegression(
        solver=backward.best_params_["solver"],
        max_iter=10000,
        C=backward.best_params_["C"],
        penalty=backward.best_params_["penalty"]
    )
    model_f=forward_model.fit(forward_x, y_train)
    model_b=backward_model.fit(backward_x, y_train)
    y_pred_b = model_b.predict(backward_x_test)
    y_pred_f= model_f.predict(forward_x_test)

    accuracy_b = accuracy_score(y_test, y_pred_b)
    precision_b = precision_score(y_test, y_pred_b)
    recall_b = recall_score(y_test, y_pred_b)
    F1_score_b = f1_score(y_test, y_pred_b)

    accuracy_f = accuracy_score(y_test, y_pred_f)
    precision_f = precision_score(y_test, y_pred_f)
    recall_f = recall_score(y_test, y_pred_f)
    F1_score_f = f1_score(y_test, y_pred_f)
    model_list=[
        {
            "model":"forward",
            "best_score":forward.best_score_,
            "best_params":forward.best_params_,
            "features": ','.join(forward_sfs_model.k_feature_names_),
            "no_features":len(forward_sfs_model.k_feature_names_),
            "accuracy":accuracy_f,
            "precision":precision_f,
            "recall":recall_f,
            "F1_score":F1_score_f

        },
        {
            "model":"backward",
            "best_score":backward.best_score_,
            "best_params":backward.best_params_,
            "features": ','.join(backward_sfs_model.k_feature_names_),
            "no_features":len(backward_sfs_model.k_feature_names_),
            "accuracy":accuracy_b,
            "precision":precision_b,
            "recall":recall_b,
            "F1_score":F1_score_b
        }

    ]
    df=pd.DataFrame(model_list)
    df.to_excel(os.path.join(final_params["report_path"],final_params["report_name"]))
    return forward_model, backward_model
