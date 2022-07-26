# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:59:09 2022

@author: kalin

"""
import lime
from lime import lime_tabular
import pandas as pd
from  train_model import load_model, load_data
from sklearn.ensemble import RandomForestClassifier 
import numpy as np
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.express as px
import dash_html_components as html
import statistics

#chargement du model 

model_rf = load_model()
train, test = load_data()
train_bis =train.drop(['TARGET'], axis=1)

#feature importance 
def load_feature_imp(model):
    importances = model.feature_importances_
    return importances 



def feature_imp(importances, data):
    #
    # Sort the feature importance in descending order
    #
    # sorted_indices = np.argsort(importances)[::-1]
    # sorted_indices10 = sorted_indices[:10]
    # sorted_indices10
    # plt.title('Feature Importance')
    # fig1= plt.bar(range(10), importances[sorted_indices10], align='center')
    # plt.xticks(range(10), train_FE_ssDM.columns[sorted_indices10], rotation=90)
    # plt.tight_layout()
    print('enter feature importance')
    importances_df = pd.DataFrame(importances)
    print(importances_df)
    importances_df['variables'] = data.columns[importances_df.index]
    importances_df = importances_df.sort_values(by=[0],ascending=False)
    most_imp = importances_df[:10]
    most_imp = most_imp.rename(columns={0: 'height'})
    
    # fig1 = plt.bar(most_imp['variables'], most_imp['height'])
    # plt.xticks(range(10), most_imp['variables'], rotation=90)
    
    fig = px.bar(x=most_imp['variables'], y=most_imp['height'])

    
    
    return fig 

 


    # def model_explanation_client(client,train):
    #     print(client)
    #     explainer = lime_tabular.LimeTabularExplainer(
    #         training_data=np.array(train.drop(['TARGET'], axis=1)),
    #         mode="classification",
    #         class_names= ['impayé', 'payé'],                                      
    #         feature_names=(train.drop(['TARGET'], axis=1).columns)) 
        
    #     exp = explainer.explain_instance(
    #         data_row=train.iloc[client], 
    #         predict_fn=model_rf.predict_proba
    #     )
        
    #     obj = html.Iframe(srcDoc=exp.as_html())
        
    #     return obj
        

def model_explanation_client(client):
    print(train_bis.columns)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_bis),
        mode="classification",
        class_names= ['impayé', 'payé'],                                      
        feature_names=train_bis )
    
    exp = explainer.explain_instance(
        data_row=train_bis.iloc[client], 
        predict_fn=model_rf.predict_proba
    )
    
    obj = html.Iframe(srcDoc=exp.as_html(),
                      width='100%',
                      height='400px')
    
    return obj
#model_expalnation_client(4,train)


# from IPython.display import display, HTML
 
# display(HTML('test'))
    
    
    
# #Description du client sur les variables les plus importantes

def var_most_imp_client(client, train_FE_ssDM):
    train_FE_ssDM['INCOME_Working'] = ['Yes' if x ==1 else 'No' for x in train_FE_ssDM['NAME_INCOME_TYPE_Working']]
    train_FE_ssDM['OWN_CAR'] = ['Yes' if x ==1 else 'No' for x in train_FE_ssDM['FLAG_OWN_CAR']]
    train_FE_ssDM['Higher education'] = ['Yes' if x ==1 else 'No' for x in train_FE_ssDM['NAME_EDUCATION_TYPE_Higher education']]
    train_FE_ssDM['GENDER'] = ['Homme' if x ==1 else 'Femme' for x in train_FE_ssDM['CODE_GENDER']]
    ext_source = round(train_FE_ssDM['EXT_SOURCE_2'].iloc[client]*100,2)
    income_type =train_FE_ssDM['INCOME_Working'].iloc[client]
    max_delay = train_FE_ssDM['INSTAL_DPD_MAX'].iloc[client]
    own_car = train_FE_ssDM['OWN_CAR'].iloc[client]
    highest_educ_level = train_FE_ssDM['Higher education'].iloc[client]
    gender_client = train_FE_ssDM['GENDER'].iloc[client]

    return ext_source, income_type, max_delay, own_car, highest_educ_level, gender_client
    

                           
# #description des variables quantitatives
# def desc_var_quanti(table, variable):
#     mean_var = statistics.mean(table.variable)
#     std_var = table.variable.std()
    
#     return  mean_var, std_var

# def fig_var_quanti(table, variable):
#     fig_quanti = px.histogram(table, variable)
#     return fig_quanti 

# #Description des variables qualitatives 
# def desc_var_quali(variable):
#     freq = variable.value_counts()
#     return freq
    
