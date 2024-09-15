import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle 
import streamlit as st

# Load Trained Model
model = tf.keras.models.load_model('model.h5')

# Load Encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoding_geo.pkl', 'rb') as file:
    oneHot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# User Input --> User will add data on Streamlit app
geography = st.selectbox('Geography', oneHot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=0.0)
estimated_salary = st.number_input('Estimated Salary', value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_product = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Input Data --> Model Prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_product],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode the Categorical Columns
# Geography and Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])  # Fixed

geo_OHE = oneHot_encoder_geo.transform(input_data[['Geography']])
geo_OHE = pd.DataFrame(geo_OHE, columns=oneHot_encoder_geo.get_feature_names_out(['Geography']))

geo_OHE.reset_index(drop=True, inplace=True)
input_data.reset_index(drop=True, inplace=True)

# Concatenate Geo_encoder with input_data
input_data = pd.concat([input_data.drop(['Geography'], axis=1), geo_OHE], axis=1)

# Scale Data
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Getting Message for User
if prediction_prob > 0.5:
    st.write("The Customer is likely to churn.")
else:
    st.write("The Customer is not likely to churn.")
