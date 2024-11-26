import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import streamlit as st

# Read data
data = pd.read_csv("AI-Data.csv")
data2 = pd.read_csv("AI-Data.csv")


# Drop unnecessary columns
columns_to_drop = ["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", "Topic", "Semester",
                  "Relation", "ParentschoolSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"]
data = data.drop(columns=columns_to_drop)

# Encode categorical variables
for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Shuffle data
u.shuffle(data)

# Split data into features and labels
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:, 4]
feats_Train = feats[0:ind]
feats_Test = feats[(ind + 1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind + 1):len(lbls)]

# Models
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Linear Model Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}

# Function to train and evaluate models
def evaluate_model(model, feats_Train, lbls_Train, feats_Test, lbls_Test):
    model.fit(feats_Train, lbls_Train)
    lbls_pred = model.predict(feats_Test)
    acc = (lbls_pred == lbls_Test).sum() / len(lbls_Test)
    report = m.classification_report(lbls_Test, lbls_pred)
    return acc, report

# Streamlit UI
st.title("Student Data Analysis")
st.sidebar.header("Model Evaluation")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

# Evaluate selected model
if st.sidebar.button("Evaluate"):
    acc, report = evaluate_model(models[model_choice], feats_Train, lbls_Train, feats_Test, lbls_Test)
    st.subheader(f"Accuracy using {model_choice}: {round(acc, 3)}")
    st.text(report)

# Graphs section
st.sidebar.header("Visualize Data")
graph_choice = st.sidebar.selectbox(
    "Select Graph", 
    ["Marks Class Count Graph", "Marks Class Semester-wise Graph", "Marks Class Gender-wise Graph",
     "Marks Class Nationality-wise Graph", "Marks Class Grade-wise Graph", "Marks Class Section-wise Graph",
     "Marks Class Topic-wise Graph", "Marks Class Stage-wise Graph", "Marks Class Absent Days-wise Graph"]
)

if graph_choice == "Marks Class Count Graph":
    st.subheader("Marks Class Count Graph")
    axes = sb.countplot(x='Class', data=data2, order=['L', 'M', 'H'])
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Semester-wise Graph":
    st.subheader("Marks Class Semester-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='Semester', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Gender-wise Graph":
    st.subheader("Marks Class Gender-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='gender', hue='Class', data=data2, order=['M', 'F'], hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Nationality-wise Graph":
    st.subheader("Marks Class Nationality-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='NationalITy', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Grade-wise Graph":
    st.subheader("Marks Class Grade-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='GradeID', hue='Class', data=data2, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Section-wise Graph":
    st.subheader("Marks Class Section-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='SectionID', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Topic-wise Graph":
    st.subheader("Marks Class Topic-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='Topic', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Stage-wise Graph":
    st.subheader("Marks Class Stage-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='StageID', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()
elif graph_choice == "Marks Class Absent Days-wise Graph":
    st.subheader("Marks Class Absent Days-wise Graph")
    fig, axesarr = plt.subplots(1, figsize=(10, 6))
    sb.countplot(x='StudentAbsenceDays', hue='Class', data=data2, hue_order=['L', 'M', 'H'], axes=axesarr)
    plt.show()
    st.pyplot()

# Test custom input
st.sidebar.header("Test Custom Input")
if st.sidebar.button("Test Input"):
    gen = st.text_input("Enter Gender (M or F):")
    nat = st.text_input("Enter Nationality:")
    pob = st.text_input("Enter Place of Birth:")
    gra = st.selectbox("Enter Grade ID:", ["G-02", "G-04", "G-05", "G-06", "G-07", "G-08", "G-09", "G-10", "G-11", "G-12"])
    sec = st.text_input("Enter Section:")
    top = st.text_input("Enter Topic:")
    sem = st.selectbox("Enter Semester (F or S):", ["F", "S"])
    rai = st.number_input("Enter Raised Hands:", min_value=0, step=1)
    res = st.number_input("Enter Visited Resources:", min_value=0, step=1)
    ann = st.number_input("Enter Announcements Viewed:", min_value=0, step=1)
    dis = st.number_input("Enter Discussions:", min_value=0, step=1)
    sur = st.selectbox("Enter Parent Answered Survey (Y or N):", ["Y", "N"])
    sat = st.selectbox("Enter Parent School Satisfaction (Good or Bad):", ["Good", "Bad"])
    absc = st.selectbox("Enter No. of Absences (Under-7 or Above-7):", ["Under-7", "Above-7"])

    if st.button("Submit Test Input"):
        # Prepare your input as numpy array and predict using your models
        # Note: Ensure the input is preprocessed similarly to how your data was prepared
        arr = np.array([rai, res, ann, dis]).reshape(1, -1)  # Example: Use the features that correspond to the model
        model = models[model_choice]
        prediction = model.predict(arr)
        st.write(f"Prediction: {prediction[0]}")
