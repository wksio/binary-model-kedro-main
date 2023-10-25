"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.8
"""
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from mlxtend.feature_selection import SequentialFeatureSelector

def split_test_train(dt_cleansing2: pd.DataFrame, split_param: Dict, y:str) -> Tuple:
    X = dt_cleansing2.drop(y, axis=1)
    y = dt_cleansing2[y]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_param["test_size"], random_state=split_param["random_state"]
    )
    return X_train, X_test, y_train, y_test

def train_logistic(X_train: pd.DataFrame, y_train: pd.DataFrame, logistic_params: Dict) -> object:
    logistic_model = LogisticRegression(
        solver=logistic_params["solver"],
        max_iter=logistic_params["max_iter"],
    )
    logistic_model.fit(X_train, y_train)
    return logistic_model

def train_logistic_forward(X_train:pd.DataFrame,y_train:pd.DataFrame,logistic_params:Dict)->object:
    forward_selector = SequentialFeatureSelector(LogisticRegression(solver=logistic_params["solver"],
                                                                     max_iter=logistic_params["max_iter"]), k_features='best', forward=True, scoring='accuracy', cv=None)
    forward_selector.fit(X_train, y_train)
    return forward_selector

def train_logistic_backward(X_train:pd.DataFrame,y_train:pd.DataFrame,logistic_params:Dict)->object:
    backward_selector = SequentialFeatureSelector(LogisticRegression(solver=logistic_params["solver"],
                                                                      max_iter=logistic_params["max_iter"]), k_features='best', forward=False, scoring='accuracy', cv=None)
    backward_selector.fit(X_train, y_train)
    return backward_selector

def form_model_sfs(sfs_model:object, X_train: pd.DataFrame, X_test:pd.DataFrame, y_train: pd.DataFrame, logistic_params:Dict) -> Tuple:
    X_train_sfs = sfs_model.transform(X_train)
    X_test_sfs = sfs_model.transform(X_test)

    # Fit the estimator using the new feature subset
    # and make a prediction on the test data
    model=LogisticRegression(solver=logistic_params["solver"],max_iter=logistic_params["max_iter"])
    model.fit(X_train_sfs, y_train)
    return model, X_test_sfs

def model_sfs_features_report(sfs_model:object,report_path:str,name:str) -> None:
    data=pd.DataFrame.from_dict(sfs_model.get_metric_dict())
    with pd.ExcelWriter(os.path.join(report_path,name)) as writer:
        # Write each DataFrame to a separate sheet
        data.to_excel(writer, sheet_name='Featues',index=False)

def model_report(model:object, X_test: pd.DataFrame, y_test: pd.DataFrame,
                     report_path:str, report_name:str) -> None:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Finding precision and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred)
    with open(os.path.join(report_path, report_name), "w") as f:
        f.write(str(report))
        f.write("\n")
        f.write(str(cm))
        f.write("\n")
        f.write("Accuracy   :" + str(accuracy))
        f.write("\n")
        f.write("Precision :" + str(precision))
        f.write("\n")
        f.write("Recall    :" + str(recall))
        f.write("\n")
        f.write("F1-score  :" + str(F1_score))