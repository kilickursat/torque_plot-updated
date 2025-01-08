import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import csv

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
           
       return df
   except Exception as e:
       st.error(f"Error loading data: {e}")
       return None

def map_machine_parameters(machine_params_series, param_mappings):
   params = {}
   for param, aliases in param_mappings.items():
       for alias in aliases:
           if alias in machine_params_series.index:
               params[param] = machine_params_series[alias]
               break
   return params

param_mappings = {
   'n1': ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'n1', 'N1', 'N1[rpm]'],
   'n2': ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'n2', 'N2', 'N2[rpm]'],
   'M_cont_value': ['M(dauer)[kNm]', 'M(dauer) [kNm]', 'M_cont[kNm]'],
   'M_max_value': ['M(max)[kNm]', 'M_max[kNm]', 'M max[kNm]'],
   'torque_constant': ['Drehmomentumrechnung [kNm/bar]', 'TC[kNm/bar]'],
   'Pressure': ['V13_SR_ArbDr_Z'],
   'RPM': ['V13_SR_Drehz_nach_Abgl_Z']
}

dtype_dict_main = {
   'V13_SR_ArbDr_Z': 'float32',
   'V13_SR_DM_Z': 'float32', 
   'V13_SR_Drehz_nach_Abgl_Z': 'float32',
   'V15_Dehn_Weg_ges_Z': 'float32',
   'V18_GesamtKraft_STZ_Z': 'float32',
   'V34_VTgeschw_Z': 'float32'
}

dtype_dict_machine = {
   'Projekt': 'string',
   'Bezeichnung neu': 'string',
   'Baureihe': 'string',
   'DA[mm]': 'float32',
   'M(dauer)[kNm]': 'float32',
   'p(dauer)[bar]': 'float32',
   'Drehmomentumrechnung [kNm/bar]': 'float32',
   'M(max)[kNm]': 'float32',
   'p(max)[bar]': 'float32',
   'n1[1/min]': 'float32',
   'n2[1/min]': 'float32',
   'Hauptlagerlast[to]': 'float32'
}

st.title('Machine Torque Analysis')

na_option = st.selectbox('Handle Missing Values:', ['Fill with Zero', 'Drop rows', 'Keep NaN'])
uploaded_main_data = st.file_uploader("Upload main data CSV", type=["csv", "xlsx"])
uploaded_machine_list = st.file_uploader("Upload machine list CSV/Excel", type=["csv", "xlsx"])

if uploaded_main_data is not None and uploaded_machine_list is not None:
   main_df = load_data(uploaded_main_data, na_option, dtype_dict_main)
   
   if main_df is not None:
       try:
           if uploaded_machine_list.type == 'text/csv':
               machine_df = pd.read_csv(uploaded_machine_list, sep=';', dtype=dtype_dict_machine, encoding='latin1', 
                                      on_bad_lines='skip', skipinitialspace=True, quoting=csv.QUOTE_ALL)
           else:
               machine_df = pd.read_excel(uploaded_machine_list, dtype=dtype_dict_machine, engine='openpyxl')
               
           if 'V13_SR_ArbDr_Z' not in main_df.columns or 'V13_SR_Drehz_nach_Abgl_Z' not in main_df.columns:
               st.error("Required columns missing in main data.")
               st.stop()
               
           machines = machine_df['Projekt'].unique()
           selected_machine = st.selectbox('Select Machine', machines)
           machine_params_series = machine_df[machine_df['Projekt'] == selected_machine].iloc[0]
           machine_params_mapped = map_machine_parameters(machine_params_series, param_mappings)
           
           required_params = ['n1', 'torque_constant', 'M_cont_value', 'M_max_value']
           missing_params = [param for param in required_params if param not in machine_params_mapped]
           if missing_params:
               st.error(f"Missing parameters: {', '.join(missing_params)}")
               st.stop()
               
           def calculate_torque(row):
               working_pressure = row['V13_SR_ArbDr_Z']
               current_speed = row['V13_SR_Drehz_nach_Abgl_Z']
               if current_speed <= 0 or pd.isna(current_speed):
                   return 0
               elif current_speed < machine_params_mapped['n1']:
                   torque = working_pressure * machine_params_mapped['torque_constant']
               else:
                   torque = (machine_params_mapped['n1'] / current_speed) * machine_params_mapped['torque_constant'] * working_pressure
               return round(torque, 2)
           
           main_df['Calculated Torque [kNm]'] = main_df.apply(calculate_torque, axis=1)
           
           sample_fraction = st.slider("Select data sample fraction:", 0.1, 1.0, 1.0)
           if sample_fraction < 1.0:
               main_df = main_df.sample(frac=sample_fraction)
           
           fig = make_subplots(rows=2, cols=1, subplot_titles=('Torque Analysis', 'Speed Profile'),
                             vertical_spacing=0.15, row_heights=[0.7, 0.3])
                             
           fig.add_trace(go.Scatter(x=main_df['Timestamp'], y=main_df['Calculated Torque [kNm]'], 
                                  mode='lines', name='Torque'), row=1, col=1)
           fig.add_trace(go.Scatter(x=main_df['Timestamp'], y=main_df['V13_SR_Drehz_nach_Abgl_Z'], 
                                  mode='lines', name='Speed'), row=2, col=1)
                                  
           fig.add_hline(y=machine_params_mapped['M_cont_value'], line_dash="dash", line_color="red",
                        annotation_text="Continuous Torque Limit", row=1, col=1)
           fig.add_hline(y=machine_params_mapped['M_max_value'], line_dash="dash", line_color="orange",
                        annotation_text="Maximum Torque Limit", row=1, col=1)
           
           fig.update_layout(height=800, showlegend=True, 
                           title_text=f'Analysis for {selected_machine}')
           fig.update_xaxes(title_text='Timestamp', row=2, col=1)
           fig.update_yaxes(title_text='Torque [kNm]', row=1, col=1)
           fig.update_yaxes(title_text='Speed [rpm]', row=2, col=1)
           
           st.plotly_chart(fig, use_container_width=True)
           
           st.download_button(
               label="Download processed data",
               data=main_df.to_csv(index=False).encode('utf-8'),
               file_name=f'{selected_machine}_processed_data.csv',
               mime='text/csv'
           )
           
       except Exception as e:
           st.error(f"Error processing data: {e}")
           st.stop()
else:
   st.warning("Please upload both main data and machine list files.")
