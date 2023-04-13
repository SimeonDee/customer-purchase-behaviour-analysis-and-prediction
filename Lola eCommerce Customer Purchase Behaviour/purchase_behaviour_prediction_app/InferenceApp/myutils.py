import pickle
import streamlit as st
from xgboost import Booster, DMatrix
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

MODEL_PATH = '../Training V3/trained_models3'
RESULTS_PATH = '../Training V3/results3'

# Saving encoded Tokenizer model utility


def save_model_pickle(model, filename):
    try:
        pickle.dump(model, open(f'{MODEL_PATH}/{filename}', 'wb'))
        print('Saved')
    except Exception as err:
        print(err)


# Loading saved Pickle model
def load_model_pickle(filename):
    try:
        model = pickle.load(open(f'{MODEL_PATH}/{filename}', 'rb'))
        return model
    except Exception as err:
        print(err)
        return None


@st.cache_resource
def get_trained_model(algorithm):
    model = None

    if algorithm.startswith('Naive'):
        model = load_model_pickle(filename='nb_model.pickle')

    elif algorithm.startswith('Support Vector'):
        model = load_model_pickle(filename='svc_model.pickle')

    elif algorithm.startswith('Random'):
        model = load_model_pickle(filename='rf_model.pickle')

    elif algorithm.startswith('XGB'):
        # model = load_model_pickle(filename='xgb_model.pickle')
        model = Booster({'nthread': 2})
        model.load_model(f'{MODEL_PATH}/xgb_model.json')

    elif algorithm.startswith('ANN'):
        # loading keras-based ANN trained model
        model = load_model(f'{MODEL_PATH}/ann_model.h5')

    elif algorithm.startswith('All'):
        nb = load_model_pickle(filename='nb_model.pickle')
        svc = load_model_pickle(filename='svc_model.pickle')
        rf = load_model_pickle(filename='rf_model.pickle')

        xgb = Booster({'nthread': 2})
        xgb.load_model(f'{MODEL_PATH}/xgb_model.json')

        nn = load_model(f'{MODEL_PATH}/ann_model.h5')
        results = {
            'nb': nb,
            'svc': svc,
            'rf': rf,
            'xgb': xgb,
            'nn': nn
        }
        return results

    return model


# loading sclale data
def load_scale_data():
    return pd.read_csv(f'{RESULTS_PATH}/scale_data_df.csv', index_col=0)

# getting code for each model


def get_model_code(algorithm):
    if algorithm.startswith('Naive'):
        return 'nb'
    elif algorithm.startswith('Support Vector'):
        return 'svc'
    elif algorithm.startswith('Random'):
        return 'rf'
    elif algorithm.startswith('XGB'):
        return 'xgb'
    elif algorithm.startswith('ANN'):
        return 'nn'
    else:  # default
        return 'rf'

# getting model name from code


def get_model_name_from_code(code):
    if code.startswith('nb'):
        return "Naive Bayes"
    elif code.startswith('rf'):
        return "Random Forest"
    elif code.startswith('svc'):
        return "Support Vector Classifier"
    elif code.startswith('nn'):
        return "Neural Networks"
    elif code.startswith('xgb'):
        return "XGBoost"
    else:
        return ''

# Make prediction


def make_prediction(model=None, recency=0, frequency=0, monetary=0.0, model_code='rf', col=None):
    """
        Scales and preprocesses input datapoints and make prediction using 'model'

        Args:
            model: trained model to use. An instance of one of (RF, NB, SVC, ANN, XGB )
            recency: Number of days ago since customer last patronized (days)
            frequency: Number of times customer patronized (int)
            monetary: Total money spent by customer across all transactions
            model_code: code name for trained model type
                Values:
                    'rf' for Random Forest | 
                    'nb' for Naive Baye's | 
                    'svc' for Support Vector Classifier | 
                    'nn' for Neural Networks | 
                    'xgb' for XGBoost |
                    'all' for all trained models

    """

    # scaling the data collected
    scale_data = load_scale_data()
    scaled_recency = (
        recency - scale_data.mini[0]) / (scale_data.maxi[0] - scale_data.mini[0])
    scaled_frequency = (
        frequency - scale_data.mini[1]) / (scale_data.maxi[1] - scale_data.mini[1])
    scaled_monetary = (
        monetary - scale_data.mini[2]) / (scale_data.maxi[2] - scale_data.mini[2])

    temp = pd.DataFrame(np.array([
        recency, frequency, monetary]).reshape(1, -1),
        columns=['Recency', 'Frequency', 'Monetary'])

    input_data = pd.DataFrame(np.array([
        scaled_recency, scaled_frequency, scaled_monetary]).reshape(1, -1),
        columns=['Recency', 'Frequency', 'Monetary'])

    col.write('Raw Input Data:')
    col.dataframe(temp)

    col.write('Scaled Input Data:')
    col.dataframe(input_data)
    prediction = None

    if model_code == 'all':
        predictions = None
        predictions = dict()

        # loading trained models

        results = get_trained_model(algorithm='All')

        nb = results['nb']
        svc = results['svc']
        rf = results['rf']
        xgb = results['xgb']
        nn = results['nn']

        # making predictions
        predictions['nb_pred'] = nb.predict_proba(input_data)
        predictions['rf_pred'] = rf.predict_proba(input_data)
        predictions['svc_pred'] = svc.predict_proba(input_data)
        # XGB
        predictions['xgb_pred'] = xgb.predict(DMatrix(input_data))
        # ANN
        predictions['nn_pred'] = nn.predict(input_data)

        return predictions

    elif model_code == 'xgb':
        prediction = model.predict(DMatrix(input_data))

    elif model_code == 'nn':
        prediction = model.predict(input_data)

    elif model_code in ['rf', 'svc', 'nb']:
        prediction = model.predict_proba(input_data)

    if prediction is None:
        st.error('Something went wrong. Sorry, Could not make prediction')

    return prediction
