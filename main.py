# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import streamlit as st

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

#loading dataset iris

iris = datasets.load_iris()

X = iris ["data"] #same as X=iris.data
y = iris ["target"]  #same y=iris.target

# train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

#Training different models
lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_= SVC()

lin_reg_fit = lin_reg.fit(X_train, y_train)   #training
log_reg_fit = log_reg.fit(X_train, y_train)
svc_fit = svc_.fit(X_train, y_train)

# creating pickle files, (save the above models)
with open("lin_reg.pkl", "wb") as li:  # wb: write mode, we are writing the doc called "li", and we want to keep there the lin_reg_fit
    pickle.dump(lin_reg_fit, li)   #pickle.dump is the function to create the file in pickle

with open("log_reg.pkl", "wb") as lo:
    pickle.dump(log_reg_fit, lo)

with open("svc_.pkl", "wb") as sv:
    pickle.dump(svc_fit, sv)

## open pickle files in main,(reading the models)

with open("lin_reg.pkl", "rb") as li:  # rb:reading mode
    linear_regression = pickle.load(li)

with open("log_reg.pkl", "rb") as lo:
    logistic_regression = pickle.load(lo)

with open("svc_.pkl", "rb") as sv:
    support_vector_classifier = pickle.load(sv)


# Function to classify the plants--> 0=setosa, 1=versicolor, 2=virginica
def classify(num):
    if num == 0:
        return (st.success("Iris Setosa"), st.image('setosa.jpg'))
    elif num == 1:
        return (st.success("Iris Versicolor"), st.image('versicolor.jpg'))
    else:
        return (st.success("Iris Virginica"), st.image('virginica.jpg'))

# create the APP after the modelling

def main():
    #sidebar title
    st.sidebar.title("Modeling IRIS by Sandra")

st.title("Modeling IRIS by Sandra")
st.subheader("User Input Parameters")


#Function for the User to input parameters in sidebar. We create the sliders
def user_input_parameter():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 6.0)  #label, min value, max value, default value
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("Sepal Length", 1.0, 6.9, 4.0)
    petal_width = st.sidebar.slider("Sepal Width", 0.1, 2.5, 1.0)
    data = {"sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width}
    features_df = pd.DataFrame(data, index=[0])
    return features_df

df = user_input_parameter()


# The User will also choose the model in a select box between 3 options. Lets create the select box

option = {"Linear Regression", "Logistic Regression", "SVM Classifier"}
model = st.sidebar.selectbox("SELECT MODEL", option)

# Add another subheader with the chosen model and the df with the input parameters.
st.subheader(model)
st.write(df)


# Create button for running the model with the function classify to show the categories and not just the numers 0 or 1 or 2.

if st.button("RUN"):
    if model == "Linear Regression":
        st.success(classify(round(linear_regression.predict(df)[0],0)))       #otherwise we will get virginica all the time because 0.90 it will fit inside the else parameters which is virginica and not versicolor, but rounded is 1, so we round it
    elif model == "Logistic Regression":
        st.success(classify(logistic_regression.predict(df)))   #here we dont need to round, because the model already gives me 1 or 2. same with svc.
    else:
        st.success(classify(support_vector_classifier.predict(df)))


if __name__ == '__main__':
    main()

