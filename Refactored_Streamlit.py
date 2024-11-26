
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Initialize Streamlit app
st.title("Marks Analysis Dashboard")
st.sidebar.header("Select an Analysis Type")

# Load the data
data_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if data_file is not None:
    data = load_data(data_file)

    # Analysis options
    options = [
        "Marks Class Count Graph",
        "Marks Class Semester-wise Graph",
        "Marks Class Gender-wise Graph",
        "Marks Class Nationality-wise Graph",
        "Marks Class Grade-wise Graph",
        "Marks Class Section-wise Graph",
        "Marks Class Topic-wise Graph",
        "Marks Class Stage-wise Graph",
        "Marks Class Absent Days-wise"
    ]

    choice = st.sidebar.selectbox("Choose an option", options)

    if choice == "Marks Class Count Graph":
        st.subheader("Marks Class Count Graph")
        fig, ax = plt.subplots()
        sb.countplot(x='Class', data=data, order=['L', 'M', 'H'], ax=ax)
        st.pyplot(fig)

    # Add more options here for other visualizations
else:
    st.warning("Please upload a CSV file to proceed.")
