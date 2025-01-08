import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Function to load the data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', parse_dates=['ts(utc)'])
    df.rename(columns={'ts(utc)': 'Timestamp'}, inplace=True)
    return df

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data loaded successfully.")
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Parameter selection
parameters = st.multiselect(
    'Choose machine parameters to analyze and visualize:',
    options=df.columns[1:],
    default=df.columns[1]
)

# Descriptive statistics
if parameters:
    st.subheader('Descriptive Statistics')
    stats = df[parameters].describe()
    st.write(stats)
else:
    st.warning('Please select at least one parameter.')

# Data visualization with subplots
if parameters:
    # Create subplots
    fig = make_subplots(rows=len(parameters), cols=1, shared_xaxes=True)
    
    for i, param in enumerate(parameters, start=1):
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df[param], mode='lines', name=param),
            row=i, col=1
        )
        # Calculate min and max for the parameter
        y_min = df[param].min()
        y_max = df[param].max()
        padding = (y_max - y_min) * 0.1  # 10% padding
        fig.update_yaxes(
            title_text=param,
            row=i, col=1,
            range=[y_min - padding, y_max + padding]
        )
    
    fig.update_layout(
        height=400 * len(parameters),  # Increased height for better visualization
        title='Sensor Data Over Time',
        showlegend=False
    )
    
    st.plotly_chart(fig)
else:
    st.warning('Please select at least one parameter to plot.')

# User instructions
st.write("### Instructions")
st.write("1. Upload your CSV file using the uploader.")
st.write("2. Select the parameters you want to analyze and visualize.")
st.write("3. The descriptive statistics and plots will update automatically.")
