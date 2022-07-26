# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:54:52 2022

@author: kalin
"""

import pandas as pd
from dash_html_components import P
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_table import DataTable
import plotly.express as px
from json import dumps, loads
from  train_model import load_model,load_data, split_data, prediction, prediction_client,train_and_benchmark_model,save_model
from  explain_model_k import feature_imp, model_explanation_client, var_most_imp_client
from tools import json_data_to_df, load_data
from sklearn.ensemble import RandomForestClassifier 
import statistics




# train_FE_ssDM, test_FE_ssDM = load_data()
# x_train, x_test, y_train, y_test = split_data(train_FE_ssDM, test_FE_ssDM)    
#model_rf = train_and_benchmark_model(x_train, x_test, y_train, y_test)
#save_model(model_rf)
train = load_data()
#train = train.drop(columns =['Unnamed: 0'])
train = train.drop(columns =['TARGET','Unnamed: 0'])
model_rf = load_model()
importances = model_rf.feature_importances_
predictions = prediction(model_rf, train)
#fig1 = explain.feature_imp(importances,train)


def callbacks(app):    
    # """Add the callbacks to the Dash app"""
    # @app.callback([Output('download_data', 'data')],
    #               [Input('submit_load_button', 'n_clicks')])
    # def load_data_dashboard(n_clicks):
    #     if n_clicks is None:
    #         raise PreventUpdate
    #     #train = load_data()
    #     print(train.shape)
    #     return dumps(train.to_dict(), default=str)
    
    
    
    @app.callback([Output('ligne_client', 'children')],
                  [Input('submit_button', 'n_clicks')],
                  [State('input_client_number', 'value')])
    def load_data_from_user(n_clicks, input_client_number):
        """Docstring to explain what the function does"""
        if n_clicks is None :
            raise PreventUpdate
        
        # chargement du fichier avec la prédiction des clients 
        df= predictions
        #df = pd.DataFrame(loads(predictions)).reset_index(drop=True)
        
        print('ici')
        if df.empty:
            raise PreventUpdate
      
        # Select data from user using 'df' & 'input_client_number'
        # df[df['user_id'] == input_client_number]
        #data_client = df[df.index == 'input_client_number']
        print(df.shape)
        score_client = round(prediction_client(input_client_number, df).loc[1],2)*100
        print('RUN PREDICTION FROM USER')
        #return score_client
        return P(f'le score client est de :{score_client}%'),
        #return DataTable(data=score_client.to_dict('rows'),
                        # columns=[{'name': i, 'id': i}
                        # for i in df.columns]),
        #prediction_client = ['pred'] # model_rf.predict_proba(data)
        # columns = [{'name': col, 'id': col} for col in score_client.columns]
        # score_client = score_client.to_dict(orient='records')
        #return score_client
    

    @app.callback([Output('Feature Importance', 'figure')],
                  [Input('show_interpretation_button', 'n_clicks')])
                  #State ('feature_fig' , 'figure')
    def feature_importance(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        print('enter callback')
        #imp = explain.load_feature_imp(saved_model)     
        fig1 = feature_imp(importances,train)
        return [fig1]
        
    
    @app.callback([Output('Explication_client', 'children')],
                  [Input('show_interpretation_button', 'n_clicks')],
                  [State('input_client_number', 'value')])
    def explication_client(n_clicks,input_client_number ):
        if n_clicks is None:
            raise PreventUpdate
        exp = model_explanation_client(input_client_number)
        return exp,
    
    @app.callback(Output('description_client', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')],
                  [State('input_client_number', 'value')])
    def description_client(n_clicks,input_client_number ):
        if n_clicks is None:
            raise PreventUpdate
            
        ext_source, income_type, max_delay, own_car, highest_educ_level, gender_client = var_most_imp_client(input_client_number, train)
        return P(f'Score source externe: {ext_source}%'), P(f'Salarié: {income_type}'),P(f'Delai maximal de remboursement credit chez Home Credit (jours): {max_delay}'),P(f'Le client possède une voiture: {own_car}'),P(f'Le client possède un haut niveau d éducation: {highest_educ_level}'), P(f'Sexe: {gender_client} '),
           
                   
    @app.callback(Output('description_all', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def description_client2(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        mean_EXT_SOURCE_2 = round(statistics.mean(train['EXT_SOURCE_2'])*100,2)
        std_EXT_SOURCE_2 = round(train['EXT_SOURCE_2'].std()*100,2)
        
                                    
        return  P(f'Moyenne: {mean_EXT_SOURCE_2}%'), P(f'Ecart-type: {std_EXT_SOURCE_2}%'),            
                               
                   
    @app.callback([Output('Histo_source_externe', 'figure')],
                  [Input('show_interpretation_button', 'n_clicks')])              
    
    def histogramme(n_clicks):        
        if n_clicks is None:
            raise PreventUpdate              
        print(train.columns)          
        fig_quanti = px.histogram(train['EXT_SOURCE_2'])
        
        return [fig_quanti]    

    
                   
    @app.callback(Output('delai_remboursement', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def description_client3(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        mean_INSTAL_DPD_MAX= round(statistics.mean(train['INSTAL_DPD_MAX']),1)
        std_INSTAL_DPD_MAX = round(train['INSTAL_DPD_MAX'].std(),1)
        
                                    
        return  P(f'Moyenne: {mean_INSTAL_DPD_MAX}'), P(f'Ecart-type: {std_INSTAL_DPD_MAX}'),                    
        

           
    @app.callback([Output('Histo_delai_remboursement', 'figure')],
                  [Input('show_interpretation_button', 'n_clicks')])              
    
    def histogramme2(n_clicks):        
        if n_clicks is None:
            raise PreventUpdate              
        print(train['INSTAL_DPD_MAX'])  
        train_instal200= train[train['INSTAL_DPD_MAX']<250]
        fig_quanti2 = px.histogram(train_instal200['INSTAL_DPD_MAX'], nbins=500)
        
        return [fig_quanti2]        



    @app.callback(Output('Pourcentage_salarié', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def salarie(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        
        freq_INCOME_Working = train['INCOME_Working'].value_counts()
        freq_INCOME_Working = round((freq_INCOME_Working.iloc[0]/291560)*100,0)
        
        return  P(f'Fréquence : {freq_INCOME_Working}%'), 


    @app.callback(Output('Pourcentage_voiture', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def voiture(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        
        freq_OWN_CAR = train['OWN_CAR'].value_counts()
        freq_OWN_CAR = round((freq_OWN_CAR.iloc[0]/291560)*100,0)
        
        return  P(f'Fréquence : {freq_OWN_CAR}%'), 
    
    @app.callback(Output('education', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def education(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        
        freq_Higher_education = train['Higher education'].value_counts()
        freq_Higher_education = round((freq_Higher_education.iloc[0]/291560)*100,0)
        
        return  P(f'Fréquence : {freq_Higher_education}%'), 
    
    @app.callback(Output('sexe', 'children'),
                  [Input('show_interpretation_button', 'n_clicks')])
    
    def sexe(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        
        freq_GENDER = train['GENDER'].value_counts()
        freq_GENDER = round((freq_GENDER.iloc[0]/291560)*100,0)
        
        return  P(f'Fréquence : {freq_GENDER}%'), 



    @app.callback(Output('output-container-range-slider', 'children'),
                  [Input('slider_age', 'value')])
    def update_output(value):
        if value is None:
            raise PreventUpdate
        return 'Echantillon séléctionné "{}" ans'.format(value)



    
    @app.callback(Output('score_ext' , 'children'),
                [Input('slider_age', 'value')])
    
    def score_ext(value):   
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        mean_EXT_SOURCE_2 = round(statistics.mean(df['EXT_SOURCE_2']),2)*100
        std_EXT_SOURCE_2 = round(df['EXT_SOURCE_2'].std(),2)*100
        
                                    
        return  P(f'Moyenne: {mean_EXT_SOURCE_2}%'), P(f'Ecart-type: {std_EXT_SOURCE_2}%'), 
    


    @app.callback([Output('Histo_source_externe2', 'figure')],
                  [Input('slider_age', 'value')])             
    
    def histogramme3(value): 
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        fig_quanti3 = px.histogram(df['EXT_SOURCE_2'], nbins = 1000)
        
        return [fig_quanti3]      
    
    
    
    @app.callback(Output('delai_remboursement2' , 'children'),
                [Input('slider_age', 'value')])
    
    def max_rem(value):
        if value is None:
            raise PreventUpdate
        df = train.loc[value]    
        mean_INSTAL_DPD_MAX= round(statistics.mean(df['INSTAL_DPD_MAX']),2)
        std_INSTAL_DPD_MAX = round(df['INSTAL_DPD_MAX'].std(),2)
        
                                    
        return  P(f'Moyenne: {mean_INSTAL_DPD_MAX}'), P(f'Ecart-type: {std_INSTAL_DPD_MAX}'), 
         

    @app.callback([Output('Histo_delai_remboursement2', 'figure')],
                  [Input('slider_age', 'value')])              
    
    def histogramme4(value): 
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        fig_quanti4 = px.histogram(df['INSTAL_DPD_MAX'], nbins=500)
        
        return [fig_quanti4]      



    @app.callback(Output('Pourcentage_salarié2', 'children'),
                  [Input('slider_age', 'value')])
    
    def salarie2(value):   
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        freq_INCOME_Working = df['INCOME_Working'].value_counts()
        freq_INCOME_Working = (freq_INCOME_Working.iloc[0]/df.shape[0])*100
        
        return  P(f'Fréquence : {freq_INCOME_Working}%'), 


    @app.callback(Output('Pourcentage_voiture2', 'children'),
                  [Input('slider_age', 'value')])
    
    def voiture2(value):
        if value is None:
            raise PreventUpdate
        df = train.loc[value]   
        freq_OWN_CAR = df['OWN_CAR'].value_counts()
        freq_OWN_CAR = round((freq_OWN_CAR.iloc[0]/df.shape[0])*100,0)
        
        return  P(f'Fréquence : {freq_OWN_CAR}%'), 
    
    @app.callback(Output('education2', 'children'),
                  [Input('slider_age', 'value')])
    
    def education2(value):
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        freq_Higher_education = df['Higher education'].value_counts()
        freq_Higher_education = round((freq_Higher_education.iloc[0]/df.shape[0])*100,0)
        
        return  P(f'Fréquence : {freq_Higher_education}%'), 
    
    @app.callback(Output('sexe2', 'children'),
                  [Input('slider_age', 'value')])
    
    def sexe2(value):
        if value is None:
            raise PreventUpdate
        df = train.loc[value]
        freq_GENDER = df['GENDER'].value_counts()
        freq_GENDER = round((freq_GENDER.iloc[0]/df.shape[0])*100,0)
        
        return  P(f'Fréquence : {freq_GENDER}%'), 



  