import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO

# Function to load the data
@st.cache_data
def load_data(file, na_option):
    df = pd.read_csv(file, sep=';', parse_dates=['ts(utc)'])
    df.rename(columns={'ts(utc)': 'Timestamp'}, inplace=True)
    
    if na_option == 'Fill with Zero':
        df.fillna(0, inplace=True)
    elif na_option == 'Drop rows':
        df.dropna(inplace=True)
    # 'Keep NaN' does nothing
    
    return df

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Handle missing values
    na_options = ['Fill with Zero', 'Drop rows', 'Keep NaN']
    na_option = st.selectbox('Handle NaN values:', na_options)
    
    try:
        # Load data with progress bar
        with st.spinner('Loading data...'):
            df = load_data(uploaded_file, na_option)
        st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
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
    
    # Export statistics as CSV
    stats_csv = stats.to_csv(index=True).encode()
    st.download_button(
        label="Download Statistics as CSV",
        data=stats_csv,
        file_name='statistics.csv',
        mime='text/csv'
    )
else:
    st.warning('Please select at least one parameter.')

# Plot customization
plot_types = ['Line', 'Scatter']
plot_type = st.selectbox('Choose plot type:', plot_types)

title = st.text_input('Plot Title', 'Sensor Data Over Time')
x_label = st.text_input('X-axis Label', 'Timestamp')
y_label = st.text_input('Y-axis Label', 'Value')

scale_options = ['Linear', 'Log']
y_scale = st.radio('Y-axis Scale:', scale_options)

# Date range filter
min_date = df['Timestamp'].min()
max_date = df['Timestamp'].max()
selected_dates = st.date_input(
    'Select date range:',
    [min_date, max_date]
)
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    df = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]

# Data visualization with subplots
if parameters:
    # Create subplots
    fig = make_subplots(rows=len(parameters), cols=1, shared_xaxes=True)
    
    for i, param in enumerate(parameters, start=1):
        if plot_type == 'Line':
            trace = go.Scatter(x=df['Timestamp'], y=df[param], mode='lines', name=param)
        else:
            trace = go.Scatter(x=df['Timestamp'], y=df[param], mode='markers', name=param)
        
        fig.add_trace(trace, row=i, col=1)
        fig.update_yaxes(title_text=y_label, row=i, col=1, type='log' if y_scale == 'Log' else 'linear')
    
    fig.update_layout(
        height=400 * len(parameters),
        title=title,
        showlegend=False
    )
    
    st.plotly_chart(fig)
    
    # Export plot as PNG
    buf = BytesIO()
    fig.write_image(buf, format='png')
    st.download_button(
        label='Download Plot as PNG',
        data=buf.getvalue(),
        file_name='plot.png',
        mime='image/png'
    )
else:
    st.warning('Please select at least one parameter to plot.')

# User instructions
st.write("### Instructions")
st.write("1. Upload your CSV file using the uploader.")
st.write("2. Select how to handle NaN values.")
st.write("3. Choose the parameters you want to analyze and visualize.")
st.write("4. Customize the plots and filters as desired.")
st.write("5. Download the statistics and plots if needed.")

# Deployment guidance and documentation would be added in a README file
