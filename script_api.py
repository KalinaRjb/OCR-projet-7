
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
from flask import request, jsonify
import json
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import cross_validate,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from  train_model import load_model,load_data, split_data
import lime
from lime import lime_tabular
import codecs


# model = load_model(...)

app = Flask(__name__)

 
train, test = load_data()
train=train.drop('TARGET', axis=1)
model_rf = load_model()

# http://127.0.0.1:5000/get_predictions/?user_id=6
@app.route('/get_predictions/', methods=['GET'])
def get_predictions():
    if 'user_id' not in request.args:
        return 'Error: No id field provided. Please specify an id.'
    idx = int(request.args['user_id'])
    user_data=train.iloc[[idx]]
    prediction_client = model_rf.predict_proba(user_data)
    
    return str(prediction_client[0][1])


    
@app.route('/get_feature_importance/', methods=['GET'])
def get_feature_importance():
    importances = model_rf.feature_importances_
    importances_df = pd.DataFrame(importances)
    importances_df['variables'] = train.columns[importances_df.index]
    importances_df = importances_df.sort_values(by=[0],ascending=False)
    most_imp = importances_df[:10]
    most_imp = most_imp.rename(columns={0: 'height'})
    
    return most_imp.to_dict()

@app.route('/explanation_client/', methods=['GET'])
def explanation_client():
    if 'user_id' not in request.args:
        return 'Error: No id field provided. Please specify an id.'
    idx = int(request.args['user_id'])
    
    
    explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(train),
    mode="classification",
    class_names= ['impayé', 'payé'],                                      
    feature_names=train )

    exp = explainer.explain_instance(
        data_row=train.iloc[idx], 
        predict_fn=model_rf.predict_proba)
    
    print('here')
    

    
    return exp.as_html()

  

if __name__ == '__main__':
    app.run()

