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
    # Clean column names - remove whitespace and special characters
    df.columns = df.columns.str.strip().str.replace('\r\n', '')
    
    # Handle known column formats from Dataset 1
    if 'Gesteinsart' in df.columns:
        numeric_cols = [col for col in df.columns if any(x in col for x in 
                      ['Dr', 'Drehz', 'Vol', 'Pos', 'Kraft', 'Weg', 'geschw'])]
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    
    # Handle known column formats from Dataset 2
    if 'ts(utc)' in df.columns:
        df['ts(utc)'] = pd.to_datetime(df['ts(utc)'], errors='coerce')
        numeric_cols = [col for col in df.columns if col != 'ts(utc)']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    
    # Handle machine specifications
    spec_cols = ['DA[mm]', 'M(dauer)[kNm]', 'M(max)[kNm]', 'p(dauer)[bar]', 'p(max)[bar]']
    for col in spec_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
    
    return df

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try reading with different settings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1', on_bad_lines='skip')
                    except:
                        df = pd.read_csv(uploaded_file, encoding='cp1252', on_bad_lines='skip')
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            
            return preprocess_tbm_data(df)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    return None

def analyze_tbm_data(df):
    """Analyze TBM-specific parameters"""
    try:
        analysis = {}
        
        # Dataset 1 specific analysis
        if 'Gesteinsart' in df.columns:
            analysis['rock_types'] = df['Gesteinsart'].dropna().value_counts().to_dict()
            if 'Bohrkopf' in df.columns:
                analysis['drilling_head_types'] = df['Bohrkopf'].dropna().value_counts().to_dict()
        
        # Dataset 2 specific analysis
        if 'V13_SR_ArbDr_Z' in df.columns:
            analysis['working_pressure'] = {
                'mean': float(df['V13_SR_ArbDr_Z'].dropna().mean()),
                'max': float(df['V13_SR_ArbDr_Z'].dropna().max())
            }
        
        return analysis
    except Exception as e:
        st.error(f"Error analyzing TBM data: {str(e)}")
        return {}

def analyze_machine_specs(df):
    try:
        # Print column names for debugging
        st.write("Available columns:", df.columns.tolist())
        
        # Find the diameter column - handle both formats
        diameter_col = next((col for col in df.columns if 'DA[mm]' in col), None)
        torque_cont_col = next((col for col in df.columns if 'M(dauer)[kNm]' in col), None)
        torque_max_col = next((col for col in df.columns if 'M(max)[kNm]' in col), None)
        
        specs = {
            'Machine Series': df['Baureihe'].dropna().unique().tolist() if 'Baureihe' in df.columns else [],
            'Diameter Range (mm)': {
                'Min': float(df[diameter_col].dropna().min()) if diameter_col else 0,
                'Max': float(df[diameter_col].dropna().max()) if diameter_col else 0
            },
            'Torque Range (kNm)': {
                'Continuous': {
                    'Min': float(df[torque_cont_col].dropna().min()) if torque_cont_col else 0,
                    'Max': float(df[torque_cont_col].dropna().max()) if torque_cont_col else 0
                },
                'Maximum': {
                    'Min': float(df[torque_max_col].dropna().min()) if torque_max_col else 0,
                    'Max': float(df[torque_max_col].dropna().max()) if torque_max_col else 0
                }
            }
        }
        return specs
    except Exception as e:
        st.error(f"Error analyzing machine specs: {str(e)}")
        return None


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
