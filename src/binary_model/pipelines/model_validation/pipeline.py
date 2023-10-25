"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=cross_validation_sfs,
                inputs=["logistic_model_backward","final_main","params:y","params:logistic_params","params:backward_cv_split"],
                outputs=None,
                name="generate_cross_validation_report_backward_selection",
        ),
        node(
                func=cross_validation_sfs,
                inputs=["logistic_model_forward","final_main","params:y","params:logistic_params","params:forward_cv_split"],
                outputs=None,
                name="generate_cross_validation_report_forward_selection",
        ),
        node(
                func=hyperparameter_tunning,
                inputs=["logistic_model_forward","X_train","y_train","params:hyper_params","params:hyper_split"],
                outputs="hyper_score_forward",
                name="logistic_hyperparameter_tunning_forward",
        ),
        node(
                func=hyperparameter_tunning,
                inputs=["logistic_model_backward","X_train","y_train","params:hyper_params","params:hyper_split"],
                outputs="hyper_score_backward",
                name="logistic_hyperparameter_tunning_backward",
        ),
        node(
                func=extract_best_params,
                inputs="hyper_score_forward",
                outputs="best_forward_params",
                name="extract_best_forward_params"
        ),
        node(
                func=extract_best_params,
                inputs="hyper_score_backward",
                outputs="best_backward_params",
                name="extract_best_backward_params"
        ),
        node(
                func=cross_validation_sfs,
                inputs=["logistic_model_backward","final_main","params:y","best_backward_params","params:backward_cv_tuned_split"],
                outputs=None,
                name="generate_cross_validation_report_tuned_backward_selection",
        ),
        node(
                func=cross_validation_sfs,
                inputs=["logistic_model_forward","final_main","params:y","best_forward_params","params:forward_cv_tuned_split"],
                outputs=None,
                name="generate_cross_validation_report_tuned_forward_selection",
        ),
        node(
                func=generate_tunning_report,
                inputs=["hyper_score_forward","params:hyper_score_forward_report"],
                outputs=None,
                name="generate_score_report_tuned_forward_selection",
        ),
        node(
                func=generate_tunning_report,
                inputs=["hyper_score_backward","params:hyper_score_backward_report"],
                outputs=None,
                name="generate_score_report_tuned_backward_selection",
        ),
        node(
                func=comparison_model,
                inputs=["hyper_score_forward","logistic_model_forward","hyper_score_backward","logistic_model_backward","X_train","y_train","X_test","y_test","params:comparison_report"],
                outputs=['final_model_forward','final_model_backward'],
                name="final_model_comparison",
        ),
        
    ])
