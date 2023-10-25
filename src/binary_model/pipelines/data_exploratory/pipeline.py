"""
This is a boilerplate pipeline 'data_exploratory'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=split_into_char_numeric,
                inputs="data_r",
                outputs=["main_df_numerical", "main_df_object"],
                name="split_into_char_numeric_node",
            ),
            node(
                func=var_explo_numeric,
                inputs=["main_df_numerical","data_r","params:y"],
                outputs="main_num_info_df",
                name="var_explo_numeric_node",
            ),
            node(
                func=var_explo_object,
                inputs=["main_df_object","data_r","params:y","params:exclude"],
                outputs="main_char_info_df",
                name="var_explo_object_node",
            ),
            node(
                func=data_shape,
                inputs=["data_r","params:y"],
                outputs="main_df_shape",
                name="var_explo_data_shape_node",
            ),
            node(
                func=var_explo_target,
                inputs=["data_r","params:y"],
                outputs="main_target_info_df",
                name="var_explo_target_node",
            ),
            node(
                func=generate_var_report,
                inputs=["main_df_shape","main_char_info_df","main_num_info_df","main_target_info_df","params:report_path","params:report_name_var_main"],
                outputs=None,
                name="generate_var_report_node",
            )
    ])
