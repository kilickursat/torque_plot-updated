import streamlit as st
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, sep=';', parse_dates=['ts(utc)'])
    df.rename(columns={'ts(utc)': 'Timestamp'}, inplace=True)
    return df

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data loaded successfully.")
else:
    st.warning("Please upload a CSV file.")
    st.stop()

parameters = st.multiselect(
    'Choose machine parameters to analyze and visualize:',
    options=df.columns[1:],
    default=df.columns[1]
)

if parameters:
    st.subheader('Descriptive Statistics')
    stats = df[parameters].describe()
    st.write(stats)
    
    fig = sp.make_subplots(rows=len(parameters), cols=1, shared_xaxes=True)
    for i, sensor in enumerate(parameters, start=1):
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[sensor], name=sensor, mode='lines'), row=i, col=1)
        fig.update_yaxes(title_text=sensor, row=i, col=1)
    fig.update_layout(height=200*len(parameters), title='Sensor Data Over Time')
    st.plotly_chart(fig)
else:
    st.warning('Please select at least one parameter.')
