"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=split_test_train,
                inputs=["final_main","params:split_param","params:y"],
                outputs=["X_train","X_test","y_train","y_test"],
                name="split_data_to_test_train",
        ),
        node(
                func=train_logistic,
                inputs=["X_train","y_train","params:logistic_params"],
                outputs="logistic_model",
                name="train_with_logistic",
        ),
        node(
                func=model_report,
                inputs=["logistic_model","X_test","y_test","params:model_report_path","params:logistic_report_name"],
                outputs=None,
                name="logistic_model_report",    
        ),
         node(
                func=train_logistic_forward,
                inputs=["X_train","y_train","params:logistic_params"],
                outputs="logistic_model_forward",
                name="train_with_logistic_forward",
        ),
        node(
                func=form_model_sfs,
                inputs=["logistic_model_forward","X_train", "X_test","y_train","params:logistic_params"],
                outputs=["logistic_model_forward_p","X_test_sfsf"],
                name="form_model_sfs_forward",
        ),
        node(
                func=model_report,
                inputs=["logistic_model_forward_p","X_test_sfsf","y_test","params:model_report_path","params:logistic_forward_report_name"],
                outputs=None,
                name="logistic_forward_model_report",    
        ),
         node(
                func=train_logistic_backward,
                inputs=["X_train","y_train","params:logistic_params"],
                outputs="logistic_model_backward",
                name="train_with_logistic_backward",
        ),
        node(
                func=form_model_sfs,
                inputs=["logistic_model_backward","X_train", "X_test","y_train","params:logistic_params"],
                outputs=["logistic_model_backward_p","X_test_sfsb"],
                name="form_model_sfs_backward",
        ),
        node(
                func=model_report,
                inputs=["logistic_model_backward_p","X_test_sfsb","y_test","params:model_report_path","params:logistic_backward_report_name"],
                outputs=None,
                name="logistic_backward_model_report",    
        ),
        node(
                func=model_sfs_features_report,
                inputs=["logistic_model_backward","params:model_report_path","params:logistic_backward_features_report"],
                outputs=None,
                name="logistic_backward_features_report",    
        ),
        node(
                func=model_sfs_features_report,
                inputs=["logistic_model_forward","params:model_report_path","params:logistic_forward_features_report"],
                outputs=None,
                name="logistic_forward_features_report",    
        ),
    ])
