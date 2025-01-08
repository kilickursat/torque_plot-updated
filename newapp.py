import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO

# Function to load the data with specified data types and parse dates
@st.cache_data
def load_data(file, na_option, dtype_dict):
    try:
        df = pd.read_csv(file, sep=';', parse_dates=['ts(utc)'], dtype=dtype_dict)
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

# Function to find the correct column name based on parameter mappings
def find_column(df, mappings):
    for col in df.columns:
        if col in mappings:
            return col
    return None

# Parameter mappings
param_mappings = {
    'Pressure': ['Pressure', 'pressure', 'p(bar)', 'p [bar]'],
    'RPM': ['RPM', 'rpm', 'n[1/min]', 'n (1/min)']
}

# File uploaders
uploaded_main_data = st.file_uploader("Upload main data CSV", type=["csv"])
uploaded_machine_list = st.file_uploader("Upload machine list CSV", type=["csv","xlsx"])

if uploaded_main_data is not None and uploaded_machine_list is not None:
    # Define data type dictionary for main data, excluding 'Timestamp'
    dtype_dict_main = {
        'Pressure': 'float32',
        'RPM': 'float32',
        # Add other columns with appropriate data types
    }
    
    # Load main data
    main_df = load_data(uploaded_main_data, na_option='Fill with Zero', dtype_dict=dtype_dict_main)
    
    if main_df is not None:
        # Define data type dictionary for machine list
        dtype_dict_machine = {
            'MachineID': 'string',
            'n1': 'float32',
            'M_cont_value': 'float32',
            'torque_constant': 'float32',
            'power': 'float32',
            # Add other columns with appropriate data types
        }
        
        # Load machine list
        machine_df = pd.read_csv(uploaded_machine_list, dtype=dtype_dict_machine)
        
        # Select machine
        machines = machine_df['MachineID'].unique()
        selected_machine = st.selectbox('Select Machine', machines)
        
        # Retrieve machine parameters
        machine_params = machine_df[machine_df['MachineID'] == selected_machine].iloc[0]
        
        # Find correct column names in the main data
        pressure_col = find_column(main_df, param_mappings['Pressure'])
        rpm_col = find_column(main_df, param_mappings['RPM'])
        
        if pressure_col is None or rpm_col is None:
            st.error("Required columns not found in the main data.")
            st.stop()
        
        # Torque calculation function
        def calculate_torque(row):
            working_pressure = row[pressure_col]
            current_speed = row[rpm_col]
            if current_speed < machine_params['n1']:
                torque = working_pressure * machine_params['torque_constant']
            else:
                torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure
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
    st.warning("Please upload both main data and machine list CSV files.")
