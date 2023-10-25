"""
This is a boilerplate pipeline 'data_cleansing'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *
from ..data_exploratory.nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=one_hot_encoding,
                inputs=["data_r","main_df_object","params:col_exclude_from_ohe","params:y"],
                outputs=["ohe_main", "ohe_df_dummy"],
                name="one_hot_encoding_node",
            ),
        node(
                func=resampling,
                inputs=["ohe_main","params:y","params:smt_random_stat"],
                outputs="smt_main",
                name="resampling_smotetomek_node",
            ),
        node(
                func=outlier_treatment,
                inputs=["main_df_numerical","smt_main","params:outlier_percentile","params:y"],
                outputs="rmo_main",
                name="outlier_treatment_node",
            ),
        node(
                func=split_into_char_numeric,
                inputs="smt_main",
                outputs=["smt_main_df_numerical", "smt_main_df_object"],
                name="split_into_char_numeric_for_smt_node",
            ),
        node(
                func=var_explo_numeric,
                inputs=["smt_main_df_numerical","smt_main","params:y"],
                outputs="smt_main_num_info_df",
                name="var_explo_numeric_for_smt_node",
            ),
        node(
                func=var_explo_object,
                inputs=["smt_main_df_object","smt_main","params:y","params:exclude_smt"],
                outputs="smt_main_char_info_df",
                name="var_explo_object_for_smt_node",
        ),
        node(
                func=data_shape,
                inputs=["smt_main","params:y"],
                outputs="smt_main_df_shape",
                name="var_explo_data_shape_for_smt_node",
        ),
        node(
                func=var_explo_target,
                inputs=["smt_main","params:y"],
                outputs="smt_main_target_info_df",
                name="var_explo_target_for_smt_node",
        ),
        node(
                func=generate_var_report,
                inputs=["smt_main_df_shape","ohe_df_dummy","smt_main_num_info_df","smt_main_target_info_df","params:report_path","params:report_name_var_smt"],
                outputs=None,
                name="generate_var_report_for_smt_node",
        )
    ])