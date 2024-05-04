import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np 


# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('housing.csv') #housing.csv
    return data

# Preprocess data
def preprocess_data(data):
    # Drop rows with missing values
    data.dropna(inplace=True)
    # Define features and target
    X = data[['total_rooms', 'total_bedrooms', 'population', 'housing_median_age', 'median_income']]
    y = data['median_house_value']
    return X, y

# Train model
@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Main function
def main():
    st.title('Nabutiti House Price Prediction App')
    
    st.write("""
    Explore Different Classifier AND DataSet # 
    Which one is the best?  
    This app predicts the **Nabutiti House Price 2024**!
    """)

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('Data loaded successfully!')

    # Preprocess data
    X, y = preprocess_data(data)
 
    # Train model
    model, X_test, y_test = train_model(X, y)

    # Sidebar
    st.sidebar.header('Input Features')

    total_rooms = st.sidebar.number_input('Total Rooms', value=1)
    total_bedrooms = st.sidebar.number_input('Total Bedrooms', value=1)
    population = st.sidebar.number_input('Population', value=1)
    housing_median_age = st.sidebar.number_input('Housing Median Age', value=1)
    median_income = st.sidebar.number_input('Median Income', value=1)

    # Make prediction
    input_data = pd.DataFrame({'total_rooms': [total_rooms],
                               'total_bedrooms': [total_bedrooms],
                               'population': [population],
                               'housing_median_age': [housing_median_age],
                               'median_income': [median_income]})
    prediction = model.predict(input_data)

    st.subheader('Prediction')
    st.write('Predicted House Price:', prediction[0])

if __name__ == '__main__':
    main()

#Ohter DataSets of different models like Iris, Diabetes, wine DataSet, & Breast Cancer 

st.subheader('DataSets')

st.sidebar.header('Explore DataSets')

dataset_name = st.sidebar.selectbox("Select DataSet", ("Iris", "Breast Cancer", "Wine DataSet", "Diabetes"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    else: 
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y 

x, y = get_dataset(dataset_name) 
st.write("Shape of Dataset", x.shape)
st.write("Number of Classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth 
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf  = RandomForestClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#Classification 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#PLOT
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

#plt.show
st.pyplot(fig)

#Input details of the app user
if "my_input" not in st.session_state:
    st.session_state["my_input1"] = ""
    st.session_state["my_input2"] = ""
    st.session_state["my_input3"] = ""
    st.session_state["my_input4"] = ""
    st.session_state["my_input5"] = ""
    
st.subheader('Insert Predictor User Details Here')    
    
my_input1 = st.text_input("FIRST NAME:", st.session_state["my_input1"])
my_input2 = st.text_input("SECOND NAME:", st.session_state["my_input2"])
my_input3 = st.text_input("EMAIL ADDRESS:", st.session_state["my_input3"])
my_input4 = st.text_input("CONTACT:", st.session_state["my_input4"])
my_input5 = st.text_input("LOCATION:", st.session_state["my_input5"])
submit = st.button("Submit")
if submit:
    st.session_state["my_input1"] = my_input1
    st.write("FIRST NAME: ", my_input1)
    
    st.session_state["my_input2"] = my_input2
    st.write("SECOND NAME: ", my_input2)
    
    st.session_state["my_input3"] = my_input3
    st.write("EMAIL ADDRESS: ", my_input3)
    
    st.session_state["my_input4"] = my_input4
    st.write("CONTACT: ", my_input4)
    
    st.session_state["my_input5"] = my_input5
    st.write("LOCATION: ", my_input5)
    
    
#Functions for Pie chart, Bar Chart, and Line Chart

# Function to preprocess the data
def preprocess_data(data):
    data = data.dropna()
    data = data.drop_duplicates()
    return data

# Function to train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)

# Function to display charts
def display_charts(y_test, predictions):
    # Pie chart
    
    st.subheader('Pie Chart Representation')  
    
    fig, ax = plt.subplots()
    labels = ['Actual', 'Predicted']
    sizes = [len(y_test[y_test<=300000]), len(y_test[y_test>300000])]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)

    # Bar chart
    
    st.subheader('Bar Chart Representation')  
      
    fig, ax = plt.subplots()
    ax.bar(['Actual', 'Predicted'], [np.mean(y_test), np.mean(predictions)])
    st.pyplot(fig)

    # Line chart
    
    st.subheader('Line Chart Representation')  
    
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Observation')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("Nabutiti Charts Representation")

    # Load data
    data = load_data()

    # Preprocess data
    data = preprocess_data(data)

    # Features and target
    X = data[['total_rooms', 'total_bedrooms', 'housing_median_age', 'population', 'median_income']]
    y = data['median_house_value']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Display evaluation metrics
    st.write("Mean squared error:", mean_squared_error(y_test, predictions))
    st.write("Coefficient of determination (R^2):", r2_score(y_test, predictions))

    # Display charts
    display_charts(y_test, predictions)

if __name__ == "__main__":
    main()

# Function to display scatter plot

def display_scatter_plot(X, y):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

    for i, feature in enumerate(X.columns):
        row = i // 2
        col = i % 2
        axes[row, col].scatter(X[feature], y, alpha=0.5)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Median House Value')

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("ScatterPlot Against Median House Value")

    # Load data
    data = load_data()

    # Preprocess data
    data = preprocess_data(data)

    # Features and target
    X = data[['total_rooms', 'total_bedrooms', 'housing_median_age', 'population', 'median_income']]
    y = data['median_house_value']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Display scatter plot
    display_scatter_plot(X_test, y_test)

    # Display evaluation metrics
    st.write("Mean squared error:", mean_squared_error(y_test, predictions))
    st.write("Coefficient of determination (R^2):", r2_score(y_test, predictions))

if __name__ == "__main__":
    main()

st.write("""This web app was developed by Mature Data Group at International University of East Africa - 
         iuea. This web app is provided to Nabutiti residents and students in Kampala - Uganda 
         who may wish to know and predict house prices of Nabutiti residential area in Kampala City.""")

st.write("""#### CopyRight@2024""")