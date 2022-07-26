# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:54:07 2022

@author: kalin
"""

from dash import Dash

from dash_html_components import Div, H1, H2, H3, Button, Br
from dash_core_components import Loading, Store, Graph, Input, RangeSlider
import dash_table

from backend_k import callbacks


# https://github.com/plotly/dash/issues/1038 lien maxime et voir tuto dash 

app = Dash(__name__)

app.layout = Div(
    children=[
        # Button('loading_data',
        #        id='submit_load_button',
        #        type='submit'),
        # Div(children=[
        #     Store(id='download_data', storage_type='memory'), #garde les données en arrière plan du dashboard pour
        #     #pouvoir être appelé dans le dashboard
        #     Store(id='user_data', storage_type='memory'),
        # ]),
        Div(
            children=[
                H1('ID client'),
                Input(id='input_client_number',
                      placeholder='client number',
                      type='number',
                      value=None),
                Button('Submit',
                       id='submit_button',
                       type='submit'),
                Br(),
                H1('Score de risque du client'),
                # Div(id='user_data') #DataTable ? -> new callback with input as data and output as datatable (children)
                #dash_table.DataTable(id='ligne_client')
                # dash_table.DataTable(id='ligne_client', data=score_client.to_dict('rows'),
                #         columns=[{'name': i, 'id': i}
                #         for i in df.columns])
                Div(id='ligne_client')
            ],
            style={'background-color': '#FFD700'}
        ),
        Div(
            children=[
                H1('Interpretabilité du modèle'),
                Button('show_interpretation',
                       id='show_interpretation_button',
                       type='submit'),
                H2('Global feature importance'),
                Loading(id = 'loading_feature_importance',
                        children = [Graph(id='Feature Importance')])
                

                # H2('Explication du modèle pour le client'),
                # Div(id='Explication_client')
            ],
            style={'background-color': '#FFFFE0'}),
        
        Div(
            children=[
                H1('Explication du modèle pour le client'),
                Loading(id = 'loading_Explication_client',
                        children = [Div(id='Explication_client', style ={'width': '100%'})])
            ],
            style={'background-color': '#FFFFE0'}),
       
        Div(
            children=[
                H1('Description du client'),
                Div(id='description_client')
            ],
            style={'background-color': '#FFFFE0'}),
        Div(
            children=[
                H1('Description de tous les clients'),
                H3('Score source externe'),
                Div(id='description_all'),
                Graph(id='Histo_source_externe'),
                H3('Delai maximal de remboursement credit chez Home Credit (jours)'),
                Div(id='delai_remboursement'),
                Graph(id='Histo_delai_remboursement'),
                H3('Pourcentage de clients salarié'),
                Div(id='Pourcentage_salarié'),
                H3('Pourcentage de clients possédant une voiture'),
                Div(id='Pourcentage_voiture'),
                H3('Haut niveau d éducation'),
                Div(id='education'),
                H3('Homme'),
                Div(id='sexe'),
            ],
                
            style={'background-color': '#FFFFE0'}), 
        Div(
            children=[
                H1('Description par tranche d âge' ),
                RangeSlider(
                            id='slider_age',
                            min=0,
                            max=100,
                            step=10,
                            marks={
                                0: '0',
                                10: '10',
                                20: '20',
                                30: '30',
                                40: '40',
                                50: '50',
                                60: '60',
                                70: '70',
                                80: '80',
                                90: '90',
                                100: '100'
                                    },
                            tooltip={'placement': 'bottom', 'always_visible': True}
                            ),
                Div(id='output-container-range-slider'),
                
                H3('Score source externe'),
                Div(id='score_ext'),
                Graph(id='Histo_source_externe2'),
                
                H3('Delai maximal de remboursement credit chez Home Credit (jours)'),
                Div(id='delai_remboursement2'),
                Graph(id='Histo_delai_remboursement2'),
                
                H3('Pourcentage de clients salarié'),
                Div(id='Pourcentage_salarié2'),
               
                H3('Pourcentage de clients possédant une voiture'),
                Div(id='Pourcentage_voiture2'),
                
                H3('Haut niveau d éducation'),
                Div(id='education2'),
                
                H3('Homme'),
                Div(id='sexe2'),
              
            ],
            style={'background-color': '#FFFFE0'}),
            ]
        )

    

callbacks(app)


if __name__ == '__main__':
    app.run_server(port=1235)
