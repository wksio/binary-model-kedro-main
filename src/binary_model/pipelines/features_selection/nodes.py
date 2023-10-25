"""
This is a boilerplate pipeline 'features_selection'
generated using Kedro 0.18.8
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from typing import Tuple, Dict, List
from statsmodels.stats.outliers_influence import variance_inflation_factor

def fs_iv_woe(data:pd.DataFrame, target:str, bins:int) -> Tuple:
    iv_thresholds = [0, 0.02, 0.1, 0.3, 0.5]
    iv_labels = ['Not Useful', 'Weak Relationship', 'Medium Strength', 'Strong Relationship']
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})      
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        # print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

    newDF['Label'] = pd.cut(newDF['IV'], bins=iv_thresholds, labels=iv_labels, right=False)
    newDF['Label'] = newDF['Label'].cat.set_categories(['Strong Relationship', 'Medium Relationship', 'Weak Relationship', 'Sus Relationship'])
    newDF.loc[newDF['IV'] > 0.5, 'Label'] = 'Sus Relationship'
    return newDF, woeDF

def fs_select_iv(newDF:pd.DataFrame, iv_select_params:list) -> list:
    good_strength_vars = newDF[newDF['Label'].isin(iv_select_params)]
    good_strength_list=good_strength_vars['Variable'].tolist()
    return good_strength_list

def fs_selectfrommodel_rf(dt_cleansing2: pd.DataFrame, y:str,selectfrommodel_rf_params:Dict) -> list:
    X=dt_cleansing2.drop(y, axis=1)
    y=dt_cleansing2[y]
    rf = SelectFromModel(RandomForestClassifier(random_state=selectfrommodel_rf_params['random_state']),max_features=selectfrommodel_rf_params['max_features'])
    rf.fit(X, y)
    selected_feat_list= X.columns[(rf.get_support())]
    selected_feat_list=selected_feat_list.tolist()
    return selected_feat_list
   
def fs_rfecv_rf(dt_cleansing2: pd.DataFrame, y:str,rfecv_rf_params:Dict) -> list:
    X=dt_cleansing2.drop(y, axis=1)
    y=dt_cleansing2[y]
    rf = RandomForestClassifier(random_state=rfecv_rf_params['random_state'])
    rfecv = RFECV(estimator=rf, step=rfecv_rf_params['step'], cv=rfecv_rf_params['cv'], scoring=rfecv_rf_params['scoring'],n_jobs=rfecv_rf_params['n_jobs'])
    rfecv.fit(X, y)
    selected_feat= X.columns[(rfecv.get_support())]
    selected_feat=selected_feat.tolist()
    return selected_feat

def fs_combine_list(*input_list:List) -> List:
    combined_list = []
    for item in input_list:
        combined_list.extend(item)
    return combined_list

def fs_corr_cal(dt_cleansing2:pd.DataFrame, y:str,unique_columns:list) -> pd.DataFrame:
    # Assuming your DataFrame is named 'data' and the target variable is 'y'
    X = dt_cleansing2.drop(y, axis=1)
    # List of items to match
    search_items =unique_columns
    X=X[search_items]
    # Calculate the correlation matrix
    correlation_matrix = X.corr()
    return correlation_matrix

def fs_select_corr(corr_matrix:pd.DataFrame,unique_columns:list,corr_threshold:float) -> list:
    # get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    # find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    selected_corr = list(filter(lambda item: item not in to_drop, unique_columns))
    return selected_corr

def fs_vif_cal(dt_cleansing2:pd.DataFrame, y:str,unique_columns:list) -> pd.DataFrame:
    # Assuming your DataFrame is named 'data' and the target variable is 'y'
    X = dt_cleansing2.drop(y, axis=1)
    # List of items to match
    search_items =unique_columns
    X=X[search_items]
    # Calculate the VIF
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def fs_select_vif(vif:pd.DataFrame,vif_threshold:float) -> list:
    # Select the features with VIF below the threshold
    selected_features = vif[vif['VIF'] < vif_threshold]['feature']
    selected_vif=list(set(selected_features.tolist()))
    return selected_vif

def fs_set_final_data(data:pd.DataFrame,selected_f:list,y:str) -> pd.DataFrame:
    final_column=selected_f+[y]
    final_data=data[final_column]
    return final_data