"""
This is a boilerplate pipeline 'data_cleansing'
generated using Kedro 0.18.9
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List
from imblearn.combine import SMOTETomek

def one_hot_encoding(dt_cleansing:pd.DataFrame,df_object:pd.DataFrame,col_exclude_from_ohe:list,y:str) -> Tuple:
    if col_exclude_from_ohe != []:
        df_object2=df_object.drop(col_exclude_from_ohe,axis=1)
    else:
        df_object2=df_object
    if df_object.empty:
        df_dummy = pd.DataFrame()
    else:
        df_encode2 = pd.get_dummies(df_object2,prefix=list(df_object2.columns),drop_first=True)
        list_col=list(df_object2.columns)
        list_col.extend(col_exclude_from_ohe)
        dt_cleansing=pd.concat([dt_cleansing,df_encode2],axis=1).drop(columns=list_col,axis=1)
        dt_cleansing[y] = dt_cleansing[y].astype("category")

        # Create an empty DataFrame
        df_dummy = pd.DataFrame(columns=['variable', 'category'])
        
        for column in df_encode2.columns:
            variable_name, category = column.rsplit('_', 1)
            df_dummy=pd.concat([df_dummy,pd.DataFrame([{'variable': variable_name, 'category': column}])],axis=0)
    
    return dt_cleansing,df_dummy

def resampling(dt_cleansing:pd.DataFrame,y:str,smt_random_stat:int) -> pd.DataFrame:
    smt = SMOTETomek(random_state=smt_random_stat)
    x_smt, y_smt = smt.fit_resample(dt_cleansing.drop(y,axis=1), dt_cleansing[y])
    dt_cleansing2=pd.concat([x_smt,y_smt],axis=1)
    return dt_cleansing2

def outlier_treatment(df_numerical:pd.DataFrame,dt_cleansing2:pd.DataFrame,outlier_percentile:int,y:str) -> pd.DataFrame:
    # Calculate the percentile value for each column
    dt_cleansing2 = dt_cleansing2.astype("float")
    percentile_values = dt_cleansing2.quantile(outlier_percentile / 100)
    catogorical_var=[y]
    target_columns= [c for c in df_numerical.columns if c not in catogorical_var]
    # Replace values exceeding the percentile with the percentile value
    for column in target_columns:
        column_values = dt_cleansing2[column]
        exceed_mask = column_values > percentile_values[column]
        dt_cleansing2.loc[exceed_mask, column] = percentile_values[column]
    return dt_cleansing2
