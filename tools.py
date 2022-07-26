import pandas as pd
from json import loads


def load_test_df():
    return pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')


def json_data_to_df(json_data):
    return pd.DataFrame(loads(json_data)).reset_index(drop=True)


def load_data():
    """Loading data"""
    train_FE_ssDM = pd.DataFrame(pd.read_csv('C:/Users/kalin/train_FE_ssDM_final.csv'))
    test_FE_ssDM = pd.DataFrame(pd.read_csv('C:/Users/kalin/test_FE_ssDM_final.csv'))
    return pd.concat([train_FE_ssDM, test_FE_ssDM])

