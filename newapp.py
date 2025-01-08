import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import csv

def calculate_whisker_and_outliers(series):
   Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
   IQR = Q3 - Q1
   lower_whisker, upper_whisker = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
   outliers = series[(series < lower_whisker) | (series > upper_whisker)]
   return lower_whisker, upper_whisker, outliers

@st.cache_data
def load_data(file, na_option, dtype_dict, encoding='utf-8', sep=';', on_bad_lines='skip'):
   try:
       if file.type == 'text/csv':
           df = pd.read_csv(file, sep=sep, parse_dates=['ts(utc)'], dtype=dtype_dict, encoding=encoding, 
                          on_bad_lines=on_bad_lines, skipinitialspace=True, quoting=csv.QUOTE_ALL)
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
           matching_cols = [col for col in machine_params_series.index if col.strip().lower() == alias.strip().lower()]
           if matching_cols:
               params[param] = machine_params_series[matching_cols[0]]
               break
   return params

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

param_mappings = {
   'n1': ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Max RPM', 'n1', 'N1', 'N1[rpm]', 'N1 [rpm]'],
   'n2': ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Min RPM', 'n2', 'N2', 'N2[rpm]', 'N2 [rpm]'],
   'M_cont_value': ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque', 'M dauer', 'Mdauer', 'M_cont', 'M(cont)', 'M_cont[kNm]', 'M_cont [kNm]'],
   'M_max_Vg1': ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 'Max Torque', 'Mmax', 'M_max', 'M max[kNm]', 'M_max [kNm]'],
   'torque_constant': ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]', 'Torque Constant', 'Torque_Constant', 'TorqueConstant', 'TC[kNm/bar]', 'TC [kNm/bar]']
}

st.title('Machine Torque Analysis')

col1, col2 = st.columns(2)
with col1:
   na_option = st.selectbox('Handle Missing Values:', ['Fill with Zero', 'Drop rows', 'Keep NaN'])
   P_max = st.number_input('Enter P_max value:', min_value=0.0, value=100.0)
   nu = 0.95

with col2:
   anomaly_threshold = st.number_input('Pressure Anomaly Threshold (bar):', min_value=0.0, value=350.0)
   x_axis_max = st.number_input('Maximum X-axis value:', min_value=0.0, value=500.0)
   sample_ratio = st.slider("Data sampling ratio:", 0.01, 1.0, 0.1, help="Reduce this if plot is slow")

uploaded_main_data = st.file_uploader("Upload main data CSV", type=["csv", "xlsx"])
uploaded_machine_list = st.file_uploader("Upload machine list CSV/Excel", type=["csv", "xlsx"])

if uploaded_main_data is not None and uploaded_machine_list is not None:
   main_df = load_data(uploaded_main_data, na_option, dtype_dict_main)
   
   if main_df is not None:
       try:
           if uploaded_machine_list.type == 'text/csv':
               machine_df = pd.read_csv(uploaded_machine_list, sep=';', encoding='latin1')
           else:
               machine_df = pd.read_excel(uploaded_machine_list)
           
           machines = machine_df['Projekt'].unique()
           selected_machine = st.selectbox('Select Machine', machines)
           machine_params_series = machine_df[machine_df['Projekt'] == selected_machine].iloc[0]
           machine_params_mapped = map_machine_parameters(machine_params_series, param_mappings)
           
           def calculate_torque(row):
               working_pressure = row['V13_SR_ArbDr_Z']
               current_speed = row['V13_SR_Drehz_nach_Abgl_Z']
               
               if pd.isna(current_speed) or current_speed < 0:
                   return 0
                   
               n2 = machine_params_mapped.get('n2', 0)
               has_adjustment_motor = n2 != 0
               
               if has_adjustment_motor:
                   if current_speed <= machine_params_mapped['n1']:
                       torque = working_pressure * machine_params_mapped['torque_constant']
                   else:
                       torque = (machine_params_mapped['n1'] / current_speed) * machine_params_mapped['torque_constant'] * working_pressure
               else:
                   torque = working_pressure * machine_params_mapped['torque_constant']
               
               return round(torque, 2)

           main_df['Calculated Torque [kNm]'] = main_df.apply(calculate_torque, axis=1)
           
           plot_df = main_df.sample(frac=sample_ratio)
           torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(plot_df['Calculated Torque [kNm]'])
           rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(plot_df['V13_SR_Drehz_nach_Abgl_Z'])
           plot_df['Is_Anomaly'] = plot_df['V13_SR_ArbDr_Z'] >= anomaly_threshold

           def M_max_Vg2(rpm):
               return np.minimum(machine_params_mapped['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

           elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params_mapped['M_max_Vg1'])
           elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params_mapped['M_cont_value'])
           rpm_curve = np.linspace(0.1, machine_params_mapped['n1'], 1000)

           fig = make_subplots(rows=1, cols=1)

           # Plot torque curves
           fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_cont],
                                  y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params_mapped['M_cont_value']),
                                  mode='lines', name='M cont Max [kNm]', line=dict(color='green', width=2)))

           fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_max],
                                  y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params_mapped['M_max_Vg1']),
                                  mode='lines', name='M max Vg1 [kNm]', line=dict(color='red', width=2)))

           fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= machine_params_mapped['n1']],
                                  y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params_mapped['n1']]),
                                  mode='lines', name='M max Vg2 [kNm]', line=dict(color='red', width=2, dash='dash')))

           y_max_vg2 = M_max_Vg2(np.array([elbow_rpm_max, elbow_rpm_cont, machine_params_mapped['n1']]))
           
           fig.add_trace(go.Scatter(x=[elbow_rpm_max, elbow_rpm_max], y=[0, y_max_vg2[0]],
                                  mode='lines', line=dict(color='purple', width=1, dash='dot'), showlegend=False))
           fig.add_trace(go.Scatter(x=[elbow_rpm_cont, elbow_rpm_cont], y=[0, y_max_vg2[1]],
                                  mode='lines', line=dict(color='orange', width=1, dash='dot'), showlegend=False))
           fig.add_trace(go.Scatter(x=[machine_params_mapped['n1'], machine_params_mapped['n1']], y=[0, y_max_vg2[2]],
                                  mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False))

           normal_data = plot_df[~plot_df['Is_Anomaly']]
           anomaly_data = plot_df[plot_df['Is_Anomaly']]
           torque_outlier_data = plot_df[plot_df['Calculated Torque [kNm]'].isin(torque_outliers)]
           rpm_outlier_data = plot_df[plot_df['V13_SR_Drehz_nach_Abgl_Z'].isin(rpm_outliers)]

           fig.add_trace(go.Scatter(x=normal_data['V13_SR_Drehz_nach_Abgl_Z'], 
                                  y=normal_data['Calculated Torque [kNm]'],
                                  mode='markers', name='Normal Data',
                                  marker=dict(color=normal_data['Calculated Torque [kNm]'], 
                                            colorscale='Viridis', size=8)))

           fig.add_trace(go.Scatter(x=anomaly_data['V13_SR_Drehz_nach_Abgl_Z'], 
                                  y=anomaly_data['Calculated Torque [kNm]'],
                                  mode='markers', 
                                  name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)',
                                  marker=dict(color='red', symbol='x', size=10)))

           fig.add_trace(go.Scatter(x=torque_outlier_data['V13_SR_Drehz_nach_Abgl_Z'],
                                  y=torque_outlier_data['Calculated Torque [kNm]'],
                                  mode='markers', name='Torque Outliers',
                                  marker=dict(color='orange', symbol='diamond', size=10)))

           fig.add_trace(go.Scatter(x=rpm_outlier_data['V13_SR_Drehz_nach_Abgl_Z'],
                                  y=rpm_outlier_data['Calculated Torque [kNm]'],
                                  mode='markers', name='RPM Outliers',
                                  marker=dict(color='purple', symbol='square', size=10)))

           fig.add_hline(y=torque_upper_whisker, line_dash="dash", line_color="gray", 
                        annotation_text="Torque Upper Whisker")
           fig.add_hline(y=torque_lower_whisker, line_dash="dot", line_color="gray", 
                        annotation_text="Torque Lower Whisker")

           fig.update_layout(
               title=f'{selected_machine} - Torque Analysis (Showing {sample_ratio*100:.1f}% of data)',
               xaxis_title='Revolution [1/min]',
               yaxis_title='Torque [kNm]',
               xaxis=dict(range=[0, x_axis_max]),
               yaxis=dict(range=[0, max(60, plot_df['Calculated Torque [kNm]'].max() * 1.1)]),
               legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
               width=1000,
               height=800,
               margin=dict(l=50, r=50, t=100, b=100)
           )

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
