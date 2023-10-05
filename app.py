
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

clustering_model = 'model.sav'
loaded_model = pickle.load(open(clustering_model, 'rb'))

pca_model = 'pca_model.pkl'
loaded_pca = joblib.load(pca_model)

std_scalar = 'std_scaler.pkl'
loaded_scalar = joblib.load(std_scalar)

df = pd.read_csv("Clustered_Data.csv")

column_names = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_TRX',
    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
    'TENURE'
]

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    "<h1 style='{}'>Customer Segmentation and Prediction</h1>".format("text-align: center;"),
    unsafe_allow_html=True
)

# with st.form("my_form"):
#     balance=st.number_input(label='Balance',step=0.001,format="%.6f")
#     balance_frequency=st.number_input(label='Balance Frequency',step=0.001,format="%.6f")
#     purchases=st.number_input(label='Purchases',step=0.01,format="%.2f")
#     oneoff_purchases=st.number_input(label='OneOff_Purchases',step=0.01,format="%.2f")
#     installments_purchases=st.number_input(label='Installments Purchases',step=0.01,format="%.2f")
#     cash_advance=st.number_input(label='Cash Advance',step=0.01,format="%.6f")
#     purchases_trx=st.number_input(label='Purc hases TRX',step=1)
#     credit_limit=st.number_input(label='Credit Limit',step=0.1,format="%.1f")
#     payments=st.number_input(label='Payments',step=0.01,format="%.6f")
#     minimum_payments=st.number_input(label='Minimum Payments',step=0.01,format="%.6f")
#     prc_full_payment=st.number_input(label='PRC Full Payment',step=0.01,format="%.6f")
#     tenure=st.number_input(label='Tenure',step=1)

#     input_data=[[balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,purchases_trx,credit_limit,payments,minimum_payments,prc_full_payment,tenure]]
#     submitted = st.form_submit_button("Submit")

with st.form("my_form"):
    # Create two columns
    col1, col2 = st.columns(2)

    # Input fields in the first column
    with col1:
        balance = st.number_input(label='Balance', step=0.001, format="%.6f")
        balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
        purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
        oneoff_purchases = st.number_input(label='OneOff_Purchases', step=0.01, format="%.2f")
        installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
        cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    # Input fields in the second column
    with col2:
        
        purchases_trx = st.number_input(label='Purchases TRX', step=1)
        credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
        payments = st.number_input(label='Payments', step=0.01, format="%.6f")
        minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
        prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
        tenure = st.number_input(label='Tenure', step=1)

    input_data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]
    submitted = st.form_submit_button("Submit")






if submitted:

    scaled_data = loaded_scalar.transform(input_data)
    pca_data = loaded_pca.transform(scaled_data)
    clust=loaded_model.predict(pca_data)[0]
    print('Data Belongs to Cluster',clust)
    st.write(f"Belongs to the Cluster, {clust}")



