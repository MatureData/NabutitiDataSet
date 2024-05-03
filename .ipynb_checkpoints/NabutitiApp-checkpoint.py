#import needed libraries 

import streamlit as st
import pandas as pd
import shap 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Nabutiti House Price Prediction App

This app predicts the **Nabutiti House Price**!
""")
st.write('---')

#load the Nabutiti House Price DataSet 
housing = datasets.load_housing.cvs()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
Y = pd.DataFrame(housing.target, columns=["median_house_value"])

#sidebar 
#header of specify input parameters 
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider('longitude', X.longitude.min(), X.longitude.max(), X.longitude.mean())
    latitude = st.sidebar.slider('latitude', X.latitude.min(), X.latitude.max(), X.latitude.mean())
    housing_median_age = st.sidebar.slider('housing_median_age', X.housing_median_age.min(), X.housing_median_age.max(), X.housing_median_age.mean())
    total_rooms = st.sidebar.slider('total_rooms ', X.total_rooms.min(), X.total_rooms.max(), X.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('total_bedrooms', X.total_bedrooms.min(), X.total_bedrooms.max(), X.total_bedrooms.mean())
    population = st.sidebar.slider('population', X.population.min(), X.population.max(), X.population.mean())
    households = st.sidebar.slider('households', X.households.min(), X.households.max(), X.households.mean())
    median_income = st.sidebar.slider('median_income', X.median_income.min(), X.median_income.max(), X.median_income.mean())
    median_house_value = st.sidebar.slider('median_house_value', X.median_house_value.min(), X.median_house_value.max(), X.median_house_value.mean())
    data = {'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms':  total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'median_house_value': median_house_value}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

#main panel

#print specified input parameters 
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

#Build Regression Model 
model = RandomForestRegressor()
model.fit(X, Y)

#apply model to make prediction 
prediction = model.predict(df)

st.header('Prediction of Median House Value')
st.write(prediction)
st.write('---')

#explaining the model's prediction using shap values 
explainer = shap.TreeExplainer(model) 
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature Importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature Importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

