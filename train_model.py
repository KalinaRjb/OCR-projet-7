from operator import mod
from sklearn.ensemble import RandomForestClassifier 
import pickle
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.model_selection import cross_validate,cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 
from collections import Counter



def load_data():
    """Loading data"""
    train_FE_ssDM = pd.DataFrame(pd.read_csv('C:/Users/kalin/train_FE_ssDM_final.csv'))
    test_FE_ssDM = pd.DataFrame(pd.read_csv('C:/Users/kalin/test_FE_ssDM_final.csv'))
    train_FE_ssDM= train_FE_ssDM.drop(columns=['Unnamed: 0'])
    test_FE_ssDM= test_FE_ssDM.drop(columns=['Unnamed: 0'])
    return train_FE_ssDM, test_FE_ssDM

def split_data(train_FE_ssDM, test_FE_ssDM):
    #Standardisation des données
    sc = StandardScaler()
    x_testfinal = sc.fit_transform(test_FE_ssDM)
    x_testfinal = pd.DataFrame(x_testfinal)
    x = sc.fit_transform(train_FE_ssDM.drop('TARGET', axis=1))
    y = train_FE_ssDM[['TARGET']]

    #Premier train / test split avant reequilibrage, pour l’instant on ne fera rien avec le test
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
    return x_train,x_test,y_train,y_test 


def train_and_benchmark_model(x_train=None, x_test=None, y_train=None, y_test=None):
    # return RandomForestClassifier()

    #Rééquilibrage des données , surechantillonage avec SMOTE

    #methode de sur echantillonage
    sm = SMOTE(random_state=42)
    def sur_ech(x_train, y_train): 
        # sur echantillonage
        x_train_Sur, y_train_Sur = sm.fit_resample(x_train, y_train)

        print("Before oversampling: ",Counter(y_train))
        print("After oversampling: ",Counter(y_train_Sur))
        return x_train_Sur, y_train_Sur

    #suréchantillonage

    x_train_Sur, y_train_Sur = sur_ech(x_train, y_train)

    x_train_Sur_df = pd.DataFrame(x_train_Sur)
    y_train_Sur_df = pd.DataFrame(y_train_Sur)

    #StatifiedKfold(crossavlidation) et Random forest

    skf = StratifiedKFold(n_splits=5)
    Random_forest = RandomForestClassifier(n_estimators=50, max_depth=5)

    #with tqdm(total=len(RandomForest)*5) as pbar: #barre pour le temps d'execution
        
        
    for train_index, valid_index in skf.split(x_train_Sur_df, y_train_Sur_df):
        
        x_train_strat, x_val_strat  = x_train_Sur_df.iloc[train_index], x_train_Sur_df.iloc[valid_index]
        y_train_strat, y_val_strat  = y_train_Sur_df.iloc[train_index], y_train_Sur_df.iloc[valid_index]
    
        
        Random_forest.fit(x_train_strat, np.ravel(y_train_strat))
        y_pred = Random_forest.predict(x_val_strat)
        cnf_matrix = metrics.confusion_matrix(y_val_strat, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, 
                                      display_labels=Random_forest.classes_)
        results = classification_report(y_val_strat, y_pred, labels=[0,1])
    print(results)
    disp.plot()
    plt.show()
    
    return Random_forest

def prediction(model, x_test):
    prediction_table = model.predict_proba(x_test)
    prediction_table = pd.DataFrame(prediction_table)
    return prediction_table




def prediction_client(id_client, prediction_table):
   client = prediction_table.iloc[id_client]
   print(client)
   return client



def save_model(model):
    output = open('classifier.pkl', 'wb')
    pickle.dump(model, output)
    output.close()


def load_model():
    f = open('classifier.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    return model


if __name__ == '__main__':
    train_FE_ssDM = load_data()
    #x_train, x_test, y_train, y_test = split_data(train_FE_ssDM, test_FE_ssDM)
    model_rf = train_and_benchmark_model(x_train, x_test, y_train, y_test)
    #model_rf = train_and_benchmark_model()
    model_rf_saved = save_model(model_rf)
    table = prediction(model_rf, x_test)
    prediction_client= prediction_client(0,table)
    


