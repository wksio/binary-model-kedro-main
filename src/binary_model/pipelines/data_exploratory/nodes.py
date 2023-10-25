"""
This is a boilerplate pipeline 'data_exploratory'
generated using Kedro 0.18.8
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple

def split_into_char_numeric(main_data:pd.DataFrame) -> Tuple:
    df_numerical = main_data.select_dtypes(include='float64')
    df_object = main_data.select_dtypes(include='object')
    return df_numerical, df_object

def var_explo_numeric(df_numerical:pd.DataFrame,main_data:pd.DataFrame, y: str) -> pd.DataFrame:
    # to create new dataframe with numerical datatypes
    if y in df_numerical.columns:
        df_numerical = df_numerical.drop(columns=[y])
    num_info_df = pd.DataFrame(columns=['Column', 'TOTAL_OBS', 'MISSING_COUNT', 'MISSING_PERCENTAGE', 'MIN_VALUE', 'MEDIAN',
                                'MEAN', 'MAX_VALUE', 'MODE', 'STD_DEV', 'PCTL_P001', 'PCTL_P002', 'PCTL_P003',
                                'PCTL_P004', 'PCTL_P005', 'PCTL_P90', 'PCTL_P95', 'PCTL_P96', 'PCTL_P97', 'PCTL_P98',
                                'PCTL_P99', 'PCTL_P99_9'])

    # Iterate over each column
    for column in df_numerical.columns:
        column_data = main_data[column]
        total_obs = len(column_data)
        missing_count = column_data.isna().sum()
        
        missing_percentage = (missing_count / total_obs) * 100
        min_value = column_data.min()
        median = column_data.median()
        mean = column_data.mean()
        max_value = column_data.max()
        mode = column_data.mode().to_list()
        std_dev = column_data.std()
        percentiles = np.percentile(column_data.dropna(), [0.1, 0.2, 0.3, 0.4, 0.5, 90, 95, 96, 97, 98, 99, 99.9])
        pctl_p001, pctl_p002, pctl_p003, pctl_p004, pctl_p005, pctl_p90, pctl_p95, pctl_p96, pctl_p97, pctl_p98, pctl_p99, pctl_p99_9 = percentiles
        
        # Add the information to the DataFrame
        num_info_df=pd.concat([num_info_df,pd.DataFrame([{'Column': column, 'TOTAL_OBS': total_obs, 'MISSING_COUNT': missing_count,
                                'MISSING_PERCENTAGE': missing_percentage, 'MIN_VALUE': min_value, 'MEDIAN': median,
                                'MEAN': mean, 'MAX_VALUE': max_value, 'MODE': mode, 'STD_DEV': std_dev,
                                'PCTL_P001': pctl_p001, 'PCTL_P002': pctl_p002, 'PCTL_P003': pctl_p003,
                                'PCTL_P004': pctl_p004, 'PCTL_P005': pctl_p005, 'PCTL_P90': pctl_p90,
                                'PCTL_P95': pctl_p95, 'PCTL_P96': pctl_p96, 'PCTL_P97': pctl_p97,
                                'PCTL_P98': pctl_p98, 'PCTL_P99': pctl_p99, 'PCTL_P99_9': pctl_p99_9}])], ignore_index=True)
    return num_info_df

def var_explo_object(df_object:pd.DataFrame,main_data:pd.DataFrame, y: str, exclude: list) -> pd.DataFrame:
    if exclude:
        df_object = df_object.drop(exclude, axis=1)
    if y in df_object.columns:
        df_object = df_object.drop(columns=[y])
    char_info_df=pd.DataFrame(columns=['Column','Category','Freq'])
    # Iterate over each column
    for column in df_object.columns:
        categories = main_data[column].unique()
        for cat in categories:
            count=main_data[column].value_counts()[cat]
            freq=count
            char_info_df=pd.concat([char_info_df,pd.DataFrame([[column,cat,freq]],columns=['Column','Category','Freq'])])
    return char_info_df

def data_shape(main_data:pd.DataFrame, y: str) -> pd.DataFrame:
    df_shape=pd.DataFrame([{'row':main_data.shape[0],'col':main_data.shape[1]}])
    return df_shape

def var_explo_target(main_data:pd.DataFrame, y: str) -> pd.DataFrame:
    target_info_df=pd.DataFrame(columns=['Column','Category','Freq'])
    categories = main_data[y].unique()
    for cat in categories:
        count=main_data[y].value_counts()[cat]
        freq=count
        target_info_df=pd.concat([target_info_df,pd.DataFrame([[y,cat,freq]],columns=['Column','Category','Freq'])])
    return target_info_df

def generate_var_report(df_shape:pd.DataFrame, char_info_df:pd.DataFrame, num_info_df:pd.DataFrame, target_info_df:pd.DataFrame, report_path:str, name:str) -> None:
    # Create an ExcelWriter object
    with pd.ExcelWriter(os.path.join(report_path,name)) as writer:
        # Write each DataFrame to a separate sheet
        df_shape.to_excel(writer, sheet_name='Data_Shape',index=False)
        target_info_df.to_excel(writer,sheet_name='Target_Var',index=False)
        num_info_df.to_excel(writer, sheet_name='Num_Var',index=False)
        char_info_df.to_excel(writer, sheet_name='Char_Var',index=False)