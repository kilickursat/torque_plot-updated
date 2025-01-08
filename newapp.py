import streamlit as st
import pandas as pd
import plotly.express as px

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', parse_dates=['ts(utc)'])
    df.rename(columns={'ts(utc)': 'Timestamp'}, inplace=True)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data loaded successfully.")
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Select parameters
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

# Plot options
plot_type = st.selectbox(
    'Select plot type:',
    options=['Line Plot', 'Scatter Plot']
)

x_axis = st.selectbox(
    'Select x-axis:',
    options=['Timestamp']
)

# Generate plot
if plot_type and parameters:
    if plot_type == 'Line Plot':
        fig = px.line(df, x=x_axis, y=parameters, title='Parameter Trends Over Time')
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(df, x=x_axis, y=parameters, title='Parameter Scatter Plot Over Time')
    st.plotly_chart(fig)
else:
    st.warning('Please select plot type and parameters.')

# Instructions
st.write("### Instructions")
st.write("1. Upload your CSV file using the uploader.")
st.write("2. Select the parameters you want to analyze and visualize.")
st.write("3. Choose the plot type and x-axis for visualization.")
