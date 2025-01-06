import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import openpyxl
import json
import base64

st.set_page_config(layout="wide")

def preprocess_tbm_data(df):
    """Specific preprocessing for TBM datasets"""
    # Handle known column formats from Dataset 1 (BN G 20 600.csv)
    if 'Gesteinsart' in df.columns:
        # Convert known numeric columns with European format
        numeric_cols = [col for col in df.columns if any(x in col for x in 
                      ['Dr', 'Drehz', 'Vol', 'Pos', 'Kraft', 'Weg', 'geschw'])]
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Handle known column formats from Dataset 2 (data 21.csv)
    if 'ts(utc)' in df.columns:
        df['ts(utc)'] = pd.to_datetime(df['ts(utc)'])
        numeric_cols = [col for col in df.columns if col != 'ts(utc)']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Handle machine specifications (MMBaureuhenliste.xlsx)
    if 'Baureihe' in df.columns:
        diameter_col = 'DA[mm]\r\n'
        torque_cols = ['M(dauer)[kNm]\r\n', 'M(max)[kNm]\r\n']
        pressure_cols = ['p(dauer)[bar]\r\n', 'p(max)[bar]\r\n']
        
        for col in [diameter_col] + torque_cols + pressure_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
    
    return df

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try multiple encodings
                encodings = ['utf-8', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            
            return preprocess_tbm_data(df)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    return None

def analyze_tbm_data(df):
    """Analyze TBM-specific parameters"""
    analysis = {}
    
    # Dataset 1 specific analysis
    if 'Gesteinsart' in df.columns:
        analysis['rock_types'] = df['Gesteinsart'].value_counts().to_dict()
        analysis['drilling_head_types'] = df['Bohrkopf'].value_counts().to_dict()
    
    # Dataset 2 specific analysis
    if 'V13_SR_ArbDr_Z' in df.columns:
        analysis['working_pressure'] = {
            'mean': float(df['V13_SR_ArbDr_Z'].mean()),
            'max': float(df['V13_SR_ArbDr_Z'].max())
        }
    
    return analysis

def plot_tbm_parameters(df, selected_columns, window_size):
    n_plots = len(selected_columns)
    fig = make_subplots(rows=n_plots, cols=1, 
                       subplot_titles=selected_columns,
                       vertical_spacing=0.05, 
                       height=300*n_plots)
    
    time_col = None
    if 'ts(utc)' in df.columns:
        time_col = 'ts(utc)'
    elif 'Relativzeit' in df.columns:
        time_col = 'Relativzeit'
    
    for i, col in enumerate(selected_columns, 1):
        x_values = df.index if time_col is None else df[time_col]
        
        # Original data
        fig.add_trace(
            go.Scatter(x=x_values, y=df[col], 
                      name=f'{col} - Raw',
                      line=dict(color='blue', width=1)),
            row=i, col=1
        )
        
        # Rolling mean
        if window_size > 1:
            rolling_mean = df[col].rolling(window=window_size).mean()
            fig.add_trace(
                go.Scatter(x=x_values, y=rolling_mean, 
                          name=f'{col} - Rolling Mean ({window_size})',
                          line=dict(color='red', width=2)),
                row=i, col=1
            )
    
    fig.update_layout(height=300*n_plots + 100, showlegend=True)
    return fig

def analyze_machine_specs(df):
    specs = {
        'Machine Series': df['Baureihe'].unique().tolist(),
        'Diameter Range (mm)': {
            'Min': float(df['DA[mm]\r\n'].min()),
            'Max': float(df['DA[mm]\r\n'].max())
        },
        'Torque Range (kNm)': {
            'Continuous': {
                'Min': float(df['M(dauer)[kNm]\r\n'].min()),
                'Max': float(df['M(dauer)[kNm]\r\n'].max())
            },
            'Maximum': {
                'Min': float(df['M(max)[kNm]\r\n'].min()),
                'Max': float(df['M(max)[kNm]\r\n'].max())
            }
        }
    }
    return specs

def export_data(df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True)
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    return href

# Main App
def main():
    st.title("TBM Data Analysis Dashboard")
    
    # File upload
    col1, col2 = st.columns(2)
    with col1:
        raw_data_file = st.file_uploader("Upload TBM sensor data", type=['csv', 'xlsx', 'xls'])
    with col2:
        spec_file = st.file_uploader("Upload machine specifications", type=['csv', 'xlsx', 'xls'])
    
    # Load and analyze data
    raw_data = load_data(raw_data_file) if raw_data_file else None
    spec_data = load_data(spec_file) if spec_file else None
    
    if raw_data is not None:
        st.header("TBM Data Analysis")
        
        # Dataset identification and specific analysis
        analysis_results = analyze_tbm_data(raw_data)
        if analysis_results:
            st.subheader("Dataset Overview")
            st.json(analysis_results)
        
        # Column selection based on dataset type
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
        selected_columns = st.multiselect("Select parameters for analysis", numeric_cols)
        
        if selected_columns:
            window_size = st.slider("Select rolling window size", 1, 100, 10)
            
            # Plot selected parameters
            fig = plot_tbm_parameters(raw_data, selected_columns, window_size)
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            if st.button("Export Analysis"):
                st.markdown(export_data(raw_data[selected_columns], "tbm_analysis.xlsx"), 
                          unsafe_allow_html=True)
    
    if spec_data is not None:
        st.header("Machine Specifications Analysis")
        specs = analyze_machine_specs(spec_data)
        st.json(specs)
        
        if st.button("Export Specifications"):
            st.markdown(export_data(spec_data, "machine_specs.xlsx"), 
                      unsafe_allow_html=True)

if __name__ == "__main__":
    main()
