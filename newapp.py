import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import csv

# Function to load the data with specified data types and parse dates
@st.cache_data
def load_data(file, na_option, dtype_dict, encoding='utf-8', sep=';', on_bad_lines='skip'):
    try:
        if file.type == 'text/csv':
            df = pd.read_csv(file, sep=sep, parse_dates=['ts(utc)'], dtype=dtype_dict, encoding=encoding, on_bad_lines=on_bad_lines, skipinitialspace=True, quoting=csv.QUOTE_ALL)
        elif file.type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            df = pd.read_excel(file, parse_dates=['ts(utc)'], dtype=dtype_dict, engine='openpyxl')
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        df.rename(columns={'ts(utc)': 'Timestamp'}, inplace=True)
        
        if na_option == 'Fill with Zero':
            df.fillna(0, inplace=True)
        elif na_option == 'Drop rows':
            df.dropna(inplace=True)
        # 'Keep NaN' does nothing
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to map machine parameters based on possible column names
def map_machine_parameters(machine_params_series, param_mappings):
    params = {}
    for param, aliases in param_mappings.items():
        for alias in aliases:
            if alias in machine_params_series.index:
                params[param] = machine_params_series[alias]
                break
    return params

# Parameter mappings
param_mappings = {
    'n1': ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Max RPM', 'n1', 'N1', 'N1[rpm]', 'N2 [rpm]'],
    'n2': ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Min RPM', 'n2', 'N2', 'N2[rpm]', 'N2 [rpm]'],
    'M_cont_value': ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque',
                     'M dauer', 'Mdauer', 'M_cont', 'M(cont)', 'M_cont[kNm]', 'M_cont [kNm]'],
    'M_max_Vg1': ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 'Max Torque',
                  'Mmax', 'M_max', 'M max[kNm]', 'M_max [kNm]'],
    'torque_constant': ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]',
                       'Torque Constant', 'Torque_Constant', 'TorqueConstant', 'TC[kNm/bar]', 'TC [kNm/bar]']
}

# File uploaders
uploaded_main_data = st.file_uploader("Upload main data CSV", type=["csv", "xlsx"])
uploaded_machine_list = st.file_uploader("Upload machine list CSV", type=["csv", "xlsx"])

if uploaded_main_data is not None and uploaded_machine_list is not None:
    # Define data type dictionary for main data, excluding 'Timestamp'
    dtype_dict_main = {
        'Pressure': 'float32',
        'RPM': 'float32',
        # Add other columns with appropriate data types
    }
    
    # Load main data
    main_df = load_data(uploaded_main_data, na_option='Fill with Zero', dtype_dict=dtype_dict_main, encoding='utf-8', sep=';', on_bad_lines='skip')
    
    if main_df is not None:
        # Define data type dictionary for machine list
        dtype_dict_machine = {
            'Projekt': 'string',
            'n1': 'float32',
            'M_cont_value': 'float32',
            'torque_constant': 'float32',
            'power': 'float32',
            # Add other columns with appropriate data types
        }
        
        # Load machine list
        try:
            if uploaded_machine_list.type == 'text/csv':
                machine_df = pd.read_csv(uploaded_machine_list, sep=';', dtype=dtype_dict_machine, encoding='latin1', on_bad_lines='skip', skipinitialspace=True, quoting=csv.QUOTE_ALL)
            elif uploaded_machine_list.type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                machine_df = pd.read_excel(uploaded_machine_list, dtype=dtype_dict_machine, engine='openpyxl')
            else:
                st.error("Unsupported file type for machine list. Please upload a CSV or Excel file.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading machine list: {e}")
            st.stop()
        
        # Select machine
        machines = machine_df['Projekt'].unique()
        selected_machine = st.selectbox('Select Machine', machines)
        
        # Retrieve machine parameters
        machine_params_series = machine_df[machine_df['Projekt'] == selected_machine].iloc[0]
        
        # Map the parameters using the mappings
        machine_params_mapped = map_machine_parameters(machine_params_series, param_mappings)
        
        # Check for required parameters
        required_params = ['n1', 'torque_constant', 'M_cont_value', 'power']
        for param in required_params:
            if param not in machine_params_mapped:
                st.error(f"Missing parameter '{param}' for the selected machine.")
                st.stop()
        
        # Find correct column names in the main data
        pressure_col = None
        rpm_col = None
        for col in main_df.columns:
            if col in param_mappings['Pressure']:
                pressure_col = col
            elif col in param_mappings['RPM']:
                rpm_col = col
        if pressure_col is None or rpm_col is None:
            st.error("Required columns not found in the main data.")
            st.stop()
        
        # Torque calculation function
        def calculate_torque(row):
            working_pressure = row[pressure_col]
            current_speed = row[rpm_col]
            if current_speed < machine_params_mapped['n1']:
                torque = working_pressure * machine_params_mapped['torque_constant']
            else:
                torque = (machine_params_mapped['n1'] / current_speed) * machine_params_mapped['torque_constant'] * working_pressure
            return round(torque, 2)
        
        # Calculate torque
        main_df['Calculated Torque [kNm]'] = main_df.apply(calculate_torque, axis=1)
        
        # Optional data sampling
        sample_fraction = st.slider("Select data sample fraction for visualization:", 0.1, 1.0, 1.0)
        if sample_fraction < 1.0:
            main_df = main_df.sample(frac=sample_fraction)
        
        # Visualization
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=main_df['Timestamp'], y=main_df['Calculated Torque [kNm]'], mode='lines', name='Torque'))
        fig.update_layout(title=f'Torque Analysis for {selected_machine}', xaxis_title='Timestamp', yaxis_title='Torque [kNm]')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload both main data and machine list files.")
