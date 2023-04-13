# !pip install -r requirements.txt
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from myutils import (get_trained_model, make_prediction, get_model_code,
                     get_model_name_from_code)

INTERPRETATION = {
    '0': 'Just a Customer (Level 1)',
    '1': 'Normal Customer (Level 2)',
    '2': 'Good Customer (Level 3)',
    '3': 'High-valued Customer (Level 4)',
    '4': 'Extremely High-valued Customer (Level 5)'
}

idx = [INTERPRETATION[str(x)] for x in range(5)]
st. set_page_config(layout="wide")

st.markdown("""
    ### Welcome to Our Customer Purchase Behaviour Analysis and Prediction for an \
    eCommerce Jewelry Store using RFM and Machine Learning Approach""")

st.sidebar.title('Classify A Customer')

st.sidebar.markdown("""--- """)
algorithm = st.sidebar.selectbox('Select an Algorithm', [
    'Naive Baye\'s', 'Support Vector Classifier', 'Random Forest',
    'XGBoost', 'ANN'])  # , 'All Algorithms'])

if algorithm is not None:
    st.sidebar.markdown("""--- """)
    st.sidebar.markdown(
        '## <center><u>Customer Data Form</u></center>', unsafe_allow_html=True)
    st.sidebar.text('When last did the customer patronized you?')

    unit = 'days'
    recency = 0
    frequency = 0
    monetary = 0

    # recency
    if(st.sidebar.radio("Options", ['in days', 'as date']) == 'in days'):
        recency = st.sidebar.slider("How many days ago?", 0, 2000)
    else:
        last_date = st.sidebar.date_input("Select date of last patronage",
                                          datetime.date(datetime.now()))
        today = datetime.date(datetime.now())
        date_diff = today - last_date
        recency = date_diff.days

    # frequency
    frequency = st.sidebar.number_input(
        "How many times has customer patronized you ?",
        min_value=1, max_value=10000
    )

    # monetary
    monetary = st.sidebar.number_input(
        "How much has the customer spent in all (USD)?")

    # st.write(f'Recency: {recency} \nFrequency: {frequency} \nMonetary: {monetary} USD')
    # st.write(f'Total: {recency + frequency + monetary}')

    if(st.sidebar.button("Classify Customer")):
        st.markdown('--- ')
        st.markdown('### Classification Results')
        st.markdown('--- ')

        if algorithm == 'All':
            predictions = make_prediction(recency=recency, frequency=frequency,
                                          monetary=monetary, model_code='all')

            # result = pd.DataFrame(predictions.values(), index=predictions.keys(), columns=range(1, 6))
            # result.plot.bar()

            for key, value in predictions.items():
                print(f'Key: {key} \t Va;ue: {value}')
                algo_name = get_model_name_from_code(key)
                pred_class = np.argmax(value)
                cust_type = INTERPRETATION[str(pred_class)]
                confidence = np.round(np.max(value) * 100, 2)

                st.markdown(f"**{algo_name}'s Predictions:**")

                result = pd.DataFrame(
                    value.reshape(-1, 1), index=idx, columns=[algo_name])
                # st.write(f'Prediction:')
                # st.dataframe(result)
                plt.figure(figsize=(0.3, 0.1))
                result.plot.barh(color='purple')
                plt.xlabel('Probabilities')
                plt.ylabel('Customer type')
                st.pyplot(plt.gcf())

                st.markdown(f"***Predicted Class:*** {pred_class}")
                st.markdown(f"***Customer Type:*** {cust_type}")
                st.markdown(f"***Confidence:*** {confidence}%")
                # st.markdown(f"*Total Probs:* {value.sum()}")
                st.markdown('--- ')

        else:
            model = get_trained_model(algorithm=algorithm)

            if model is None:
                st.error(
                    'Internal Server Error: Sorry, could not load trained model')
            else:

                st.success(f'{algorithm} Model loaded successfully')

                col1, col2 = st.columns(2)

                code = get_model_code(algorithm=algorithm)
                prediction = make_prediction(
                    model=model, recency=recency,
                    frequency=frequency, monetary=monetary,
                    model_code=code, col=col1)

                if prediction is not None:
                    algo_name = get_model_name_from_code(code)
                    pred_class = np.argmax(prediction)
                    cust_type = INTERPRETATION[str(pred_class)]
                    confidence = np.round(np.max(prediction) * 100, 2)

                    col2.markdown(f"**{algo_name}'s Predictions:**")

                    result = pd.DataFrame(
                        prediction.reshape(-1, 1), index=idx, columns=[algo_name])

                    # col2.write(f'Prediction:')
                    # col2.dataframe(result)
                    plt.figure(figsize=(0.3, 0.1))
                    result.plot.barh(color='purple')
                    plt.xlabel('Probabilities')
                    plt.ylabel('Customer type')
                    col2.pyplot(plt.gcf())

                    col2.markdown(f"*Predicted Class:* {pred_class}")
                    col2.markdown(f"*Customer Type:* {cust_type}")
                    col2.markdown(f"*Confidence:* {confidence}%")
                    # col2.markdown(f"*Total Probs:* {prediction.sum()}")
                    st.markdown('--- ')
