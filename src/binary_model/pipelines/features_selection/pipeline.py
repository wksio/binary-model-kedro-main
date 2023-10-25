"""
This is a boilerplate pipeline 'features_selection'
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=fs_iv_woe,
                inputs=["rmo_main","params:y","params:iv_bins"],
                outputs=["iv_df","woe_df"],
                name="iv_woe_calculation",
        ),
        node(
                func=fs_select_iv,
                inputs=["iv_df","params:iv_select_params"],
                outputs="iv_selected",
                name="iv_selection",
        ),
        node(
                func=fs_selectfrommodel_rf,
                inputs=["rmo_main","params:y","params:selectfrommodel_rf_params"],
                outputs="selectfrommodel_rf_selected",
                name="selection_with_randomforest_selectfrommodel",
        ),
        node(
                func= fs_combine_list,
                inputs=['iv_selected', 'selectfrommodel_rf_selected'],
                outputs='combine_selected',
                name='combine_all_selection',
        ),
        node(
                func=fs_corr_cal,
                inputs=["rmo_main","params:y","combine_selected"],
                outputs="corr_matrix",
                name="calulate_corr_matrix",
        ),
        node(
                func=fs_select_corr,
                inputs=["corr_matrix","combine_selected","params:corr_threshold"],
                outputs="corr_selected",
                name="select_corr_matrix_with_threshold",
        ),
        node(
                func=fs_vif_cal,
                inputs=["rmo_main","params:y","combine_selected"],
                outputs="vif_df",
                name="calculate_vif",
        ),
        node(
                func=fs_select_vif,
                inputs=["vif_df","params:vif_threshold"],
                outputs="vif_selected",
                name="select_vif",
        ),
        node(
                func=fs_set_final_data,
                inputs=["rmo_main","corr_selected","params:y"],
                outputs="final_main",
                name="set_final_data_set",
        ),

    ])
