import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import io
import base64
from PIL import Image
import os
import tempfile

def load_machine_specs(file):
    specs_df = pd.read_excel(file)
    specs_df.columns = specs_df.columns.str.strip()  # Strip any leading/trailing whitespace and newlines
    
    # Display the columns in a more UX-friendly design
    with st.expander("Columns in the Uploaded Excel File"):
        st.dataframe(pd.DataFrame(specs_df.columns, columns=["Column Names"]))
    
    return specs_df

def get_machine_params(specs_df, machine_type):
    machine_data = specs_df[specs_df['Projekt'] == machine_type].iloc[0]
    
    def find_column(possible_names):
        for name in possible_names:
            if name in machine_data.index:
                return name
        return None

    # Define possible column names
    n1_names = ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]']
    n2_names = ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]']
    m_cont_names = ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)']
    m_max_names = ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]']
    torque_constant_names = ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]']

    # Find the correct column names
    n1_col = find_column(n1_names)
    n2_col = find_column(n2_names)
    m_cont_col = find_column(m_cont_names)
    m_max_col = find_column(m_max_names)
    torque_constant_col = find_column(torque_constant_names)

    # Return machine parameters
    return {
        'n1': machine_data[n1_col],
        'n2': machine_data[n2_col],
        'M_cont_value': machine_data[m_cont_col],
        'M_max_Vg1': machine_data[m_max_col],
        'torque_constant': machine_data[torque_constant_col]
    }

def calculate_whisker_and_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    whisker_length = 1.5 * IQR
    lower_whisker = Q1 - whisker_length
    upper_whisker = Q3 + whisker_length
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    return lower_whisker, upper_whisker, outliers

class PDF(FPDF):
    def header(self):
        # Add any header content if needed
        pass

    def footer(self):
        # Add any footer content if needed
        pass

import tempfile
import os

def create_pdf_report(df, machine_params, selected_machine, anomaly_threshold):
    pdf = PDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Torque Analysis Report - {selected_machine}", ln=True, align="C")
    pdf.ln(10)

    # Machine Parameters
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Machine Parameters", ln=True)
    pdf.set_font("Arial", "", 12)
    for key, value in machine_params.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(10)

    # Statistical Analysis
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Statistical Analysis", ln=True)
    pdf.set_font("Arial", "", 12)
    for column in ['Revolution [rpm]', 'Calculated torque [kNm]', 'Working pressure [bar]']:
        pdf.cell(0, 10, f"{column} Statistics:", ln=True)
        stats = df[column].describe()
        for stat, value in stats.items():
            pdf.cell(0, 10, f"  {stat}: {value:.2f}", ln=True)
        pdf.ln(5)

    # Anomaly Detection Results
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Anomaly Detection Results", ln=True)
    pdf.set_font("Arial", "", 12)
    anomaly_data = df[df['Is_Anomaly']]
    pdf.cell(0, 10, f"Total data points: {len(df)}", ln=True)
    pdf.cell(0, 10, f"Anomaly data points: {len(anomaly_data)}", ln=True)
    pdf.cell(0, 10, f"Percentage of anomalies: {len(anomaly_data) / len(df) * 100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Anomaly threshold: {anomaly_threshold} bar", ln=True)

    return pdf

def set_page_config():
    st.set_page_config(
        page_title="Herrenknecht Torque Analysis",
        page_icon="🚀",
        layout="wide"
    )

def add_logo():
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: url(https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png);
            background-repeat: no-repeat;
            background-size: 200px;
            padding-top: 120px;
            background-position: 20px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_background_color():
    herrenknecht_green = "#90EE90"  # Herrenknecht green color
    
    # Custom CSS to set the background color
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {herrenknecht_green};
            color: white;
        }}
        .stSidebar .sidebar-content {{
            background-color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_page_config()
    set_background_color()
    add_logo()

    st.title("Enhanced Torque Analysis App")
    st.sidebar.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

    # File uploaders
    raw_data_file = st.file_uploader("Upload Raw Data CSV", type="csv")
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")

    if machine_specs_file is None:
        st.warning("Please upload Machine Specifications XLSX file.")
        return

    try:
        machine_specs = load_machine_specs(machine_specs_file)
        machine_types = machine_specs['Projekt'].unique()
        selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

        machine_params = get_machine_params(machine_specs, selected_machine)

        # Display loaded parameters in a table
        st.write("Loaded Machine Parameters:")
        st.table(pd.DataFrame([machine_params]))

    except Exception as e:
        st.error(f"An error occurred while processing the machine specifications: {str(e)}")
        return

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    if raw_data_file is None:
        st.info("Please upload a Raw Data CSV file to begin the analysis.")
        return

    df = pd.read_csv(raw_data_file, sep=';', decimal=',')
    
    # Rename and clean columns as needed
    df = df.rename(columns={
        'AzV.V13_SR_ArbDr_Z | DB    60.DBD    26': 'Working pressure [bar]',
        'AzV.V13_SR_Drehz_nach_Abgl_Z | DB    60.DBD    30': 'Revolution [rpm]'
    })
    
    df['Revolution [rpm]'] = pd.to_numeric(df['Revolution [rpm]'], errors='coerce')
    df['Working pressure [bar]'] = pd.to_numeric(df['Working pressure [bar]'], errors='coerce')
    df = df.dropna(subset=['Revolution [rpm]', 'Working pressure [bar]'])

    # RPM Statistics
    rpm_max_value = df['Revolution [rpm]'].max()
    st.sidebar.write(f"Recommended value for x-axis based on the Max RPM in Data: {rpm_max_value:.2f}")

    # Allow user to set x_axis_max
    x_axis_max = st.sidebar.number_input("X-axis maximum", value=rpm_max_value, min_value=1.0, max_value=100.0)
    
    # Filter data points between n2 and n1 rpm
    df = df[(df['Revolution [rpm]'] >= machine_params['n2']) & (df['Revolution [rpm]'] <= machine_params['n1'])]

    # Calculate torque
    def calculate_torque_wrapper(row):
        working_pressure = row['Working pressure [bar]']
        current_speed = row['Revolution [rpm]']

        if current_speed < machine_params['n1']:
            torque = working_pressure * machine_params['torque_constant']
        else:
            torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure

        return round(torque, 2)

    df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

    # Calculate whiskers and outliers
    torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
    rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df['Revolution [rpm]'])

    # Anomaly detection
    df['Is_Anomaly'] = df['Working pressure [bar]'] >= anomaly_threshold

    # Function to calculate M max Vg2
    def M_max_Vg2(rpm):
        return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

    # Calculate the intersection points
    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

    # Generate rpm values for the continuous curves
    rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)  # Avoid division by zero

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot torque curves
    ax.plot(rpm_curve[rpm_curve <= elbow_rpm_cont],
            np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params['M_cont_value']),
            'g-', linewidth=2, label='M cont Max [kNm]')

    ax.plot(rpm_curve[rpm_curve <= elbow_rpm_max],
            np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']),
            'r-', linewidth=2, label='M max Vg1 [kNm]')

    ax.plot(rpm_curve[rpm_curve <= machine_params['n1']], M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]),
            'r--', linewidth=2, label='M max Vg2 [kNm]')

    # Add vertical lines at the elbow points
    ax.plot([elbow_rpm_max, elbow_rpm_max], [0, machine_params['M_max_Vg1']], color='purple', linestyle=':', linewidth=3)
    ax.plot([elbow_rpm_cont, elbow_rpm_cont], [0, machine_params['M_cont_value']], color='orange', linestyle=':', linewidth=3)

    # Add a truncated vertical line at n1
    ax.plot([machine_params['n1'], machine_params['n1']], [0, machine_params['M_cont_value']], color='black', linestyle='--', linewidth=2)

    # Plot calculated torque vs RPM, differentiating between normal, anomaly, and outlier points
    normal_data = df[(~df['Is_Anomaly']) & (~df['Calculated torque [kNm]'].isin(torque_outliers))]
    anomaly_data = df[df['Is_Anomaly']]
    torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers) & (~df['Is_Anomaly'])]
    rpm_outlier_data = df[df['Revolution [rpm]'].isin(rpm_outliers) & (~df['Is_Anomaly'])]

    scatter_normal = ax.scatter(normal_data['Revolution [rpm]'], normal_data['Calculated torque [kNm]'],
                                c=normal_data['Calculated torque [kNm]'], cmap='viridis',
                                s=50, alpha=0.6, label='Normal Data')
    scatter_anomaly = ax.scatter(anomaly_data['Revolution [rpm]'], anomaly_data['Calculated torque [kNm]'],
                                 color='red', s=100, alpha=0.8, marker='X', label=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)')
    scatter_torque_outliers = ax.scatter(torque_outlier_data['Revolution [rpm]'], torque_outlier_data['Calculated torque [kNm]'],
                                         color='orange', s=100,alpha=0.8, marker='D', label='Torque Outliers')
    scatter_rpm_outliers = ax.scatter(rpm_outlier_data['Revolution [rpm]'], rpm_outlier_data['Calculated torque [kNm]'],
                                      color='purple', s=100, alpha=0.8, marker='s', label='RPM Outliers')

    # Add horizontal lines for the torque whiskers
    ax.axhline(y=torque_upper_whisker, color='gray', linestyle='--', linewidth=1, label='Torque Upper Whisker')
    ax.axhline(y=torque_lower_whisker, color='gray', linestyle=':', linewidth=1, label='Torque Lower Whisker')

    # Set plot limits and labels
    ax.set_xlim(0, x_axis_max)
    ax.set_ylim(0, max(60, df['Calculated torque [kNm]'].max() * 1.1))
    ax.set_xlabel('Drehzahl / speed / vitesse / revolutiones [1/min]')
    ax.set_ylabel('Drehmoment / torque / couple / par de giro [kNm]')
    plt.title(f'{selected_machine} - Torque Analysis')

    # Add grid
    ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)

    # Add text annotations
    ax.text(elbow_rpm_max * 0.5, machine_params['M_max_Vg1'] * 1.05, f'M max (max.): {machine_params["M_max_Vg1"]} kNm',
            fontsize=10, ha='center', va='bottom', color='red')
    ax.text(elbow_rpm_cont * 0.5, machine_params['M_cont_value'] * 0.95, f'M cont (max.): {machine_params["M_cont_value"]} kNm',
            fontsize=10, ha='center', va='top', color='green')

    # Add text annotations for elbow points and n1
    ax.text(elbow_rpm_max, 0, f'{elbow_rpm_max:.2f}', ha='right', va='bottom', color='purple', fontsize=8)
    ax.text(elbow_rpm_cont, 0, f'{elbow_rpm_cont:.2f}', ha='right', va='bottom', color='orange', fontsize=8)
    ax.text(machine_params['n1'], machine_params['M_cont_value'], f'n1: {machine_params["n1"]}', ha='right', va='top', color='black', fontsize=8, rotation=90)

    # Add colorbar for the scatter plot
    cbar = plt.colorbar(scatter_normal)
    cbar.set_label('Calculated Torque [kNm]')

    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # Display statistics on the web page
    st.header("Data Statistics")
    for column in ['Revolution [rpm]', 'Calculated torque [kNm]', 'Working pressure [bar]']:
        st.subheader(f"{column} Statistics")
        st.write(df[column].describe())

    # Display anomaly detection results on the web page
    st.header("Anomaly Detection Results")
    anomaly_data = df[df['Is_Anomaly']]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Total data points: {len(df)}")
        st.write(f"Normal data points: {len(df) - len(anomaly_data)}")
    with col2:
        st.write(f"Anomaly data points: {len(anomaly_data)}")
        st.write(f"Percentage of anomalies: {len(anomaly_data) / len(df) * 100:.2f}%")
    with col3:
        st.write(f"Anomaly threshold: {anomaly_threshold} bar")

# Create and offer PDF report for download
if st.button("Generate PDF Report"):
    pdf = create_pdf_report(df, machine_params, selected_machine, anomaly_threshold)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"torque_analysis_report_{selected_machine}.pdf",
        mime="application/pdf"
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"torque_analysis_report_{selected_machine}.pdf",
        mime="application/pdf"
    )

    # Add footer with creator information
    st.markdown("---")
    st.markdown("Created by Kursat Kilic - Geotechnical Digitalization")

if __name__ == "__main__":
    main()
