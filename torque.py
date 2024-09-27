import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

def load_machine_specs(file):
    if file.name.endswith('.csv'):
        specs_df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        specs_df = pd.read_excel(file)
    else:
        st.error("Unsupported file format! Please upload a CSV or XLSX file.")
        return None

    specs_df.columns = specs_df.columns.str.strip()  # Strip any leading/trailing whitespace and newlines
    
    with st.expander("Columns in the Uploaded File"):
        st.dataframe(pd.DataFrame(specs_df.columns, columns=["Column Names"]))
    
    return specs_df

def find_column(possible_names, df_columns):
    for name in possible_names:
        if name in df_columns:
            return name
    # If exact matches are not found, try partial matches
    for name in possible_names:
        match = [col for col in df_columns if name.lower() in col.lower()]
        if match:
            return match[0]
    return None

def get_machine_params(specs_df, machine_type):
    machine_data = specs_df[specs_df['Projekt'] == machine_type].iloc[0]
    
    # Define possible column names
    n1_names = ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Speed n1']
    n2_names = ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Speed n2']
    m_cont_names = ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque']
    m_max_names = ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 'Max Torque']
    torque_constant_names = ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]', 'Torque Constant']

    # Find the correct column names
    n1_col = find_column(n1_names, machine_data.index)
    n2_col = find_column(n2_names, machine_data.index)
    m_cont_col = find_column(m_cont_names, machine_data.index)
    m_max_col = find_column(m_max_names, machine_data.index)
    torque_constant_col = find_column(torque_constant_names, machine_data.index)

    if not all([n1_col, n2_col, m_cont_col, m_max_col, torque_constant_col]):
        st.error("Required columns not found in the uploaded file. Please ensure correct data format.")
        return None

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

def set_page_config():
    st.set_page_config(
        page_title="Herrenknecht Torque Analysis",
        page_icon="https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png",
        layout="wide"
    )

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: rgb(220, 234, 197);
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_logo():
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: url(https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png);
            background-repeat: no-repeat;
            background-size: 140px;
            background-position: 10px 10px;
        }
        [data-testid="stSidebar"]::before {
            content: "";
            display: block;
            height: 100px; 
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        .sidebar-content {
            padding-top: 100px;
        }
        .sidebar-content > * {
            margin-bottom: 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    return img_str

def main():
    set_page_config()
    set_background_color()
    add_logo()

    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")
    st.sidebar.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")

    raw_data_file = st.file_uploader("Upload Raw Data", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications", type=["csv", "xlsx"])

    if machine_specs_file is not None:
        try:
            machine_specs = load_machine_specs(machine_specs_file)
            if machine_specs is None:
                return
            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

            machine_params = get_machine_params(machine_specs, selected_machine)
            if machine_params is None:
                return
            
            st.write("**Loaded Machine Parameters:**")
            params_df = pd.DataFrame([machine_params])   
            styled_table = params_df.style.set_table_styles([
                {'selector': 'th', 'props': [('border', '2px solid black'), ('padding', '5px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
                {'selector': '', 'props': [('border-collapse', 'collapse')]}
            ]).to_html()

            styled_table = styled_table.split('</style>')[-1]
            st.markdown(
                f"""
                <style>
                table {{
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 0.9em;
                    font-family: sans-serif;
                    min-width: 400px;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                }}
                table thead tr {{
                    background-color: rgb(0, 62, 37);
                    color: #ffffff;
                    text-align: left;
                }}
                table th, table td {{
                    padding: 12px 15px;
                    border: 2px solid black;
                }}
                table tbody tr {{
                    border-bottom: 1px solid #dddddd;
                }}
                table tbody tr:nth-of-type(even) {{
                    background-color: #f3f3f3;
                }}
                table tbody tr:last-of-type {{
                    border-bottom: 2px solid rgb(0, 62, 37);
                }}
                </style>
                {styled_table}
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred while processing the machine specifications: {str(e)}")
            return
    else:
        st.warning("Please upload Machine Specifications file.")
        return

    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    if raw_data_file is not None:
        if raw_data_file.name.endswith(".csv"):
            df = pd.read_csv(raw_data_file, sep=';', decimal=',')
        elif raw_data_file.name.endswith(".xlsx"):
            df = pd.read_excel(raw_data_file)
        else:
            st.error("Unsupported file format! Please upload a CSV or XLSX file.")
            return

        # Dynamic renaming for sensor columns
        revolution_names = ['Revolution [rpm]', 'Drehzahl', 'Speed', 'RPM','Drehz_nach_Abgl_Z','Drehz']
        working_pressure_names = ['Working pressure [bar]', 'Arbeitsdruck', 'Pressure', 'Bar','ArbDr_Z','ArbDr']

        revolution_col = find_column(revolution_names, df.columns)
        working_pressure_col = find_column(working_pressure_names, df.columns)

        if not revolution_col or not working_pressure_col:
            st.error("Required columns for Revolution and Working Pressure not found.")
            return

        df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')
        df[working_pressure_col] = pd.to_numeric(df[working_pressure_col], errors='coerce')
        df = df.dropna(subset=[revolution_col, working_pressure_col])

        rpm_stats = df[revolution_col].describe()
        rpm_max_value = rpm_stats['max']
        st.sidebar.write(f"Recommended value for x-axis based on the Max RPM in Data: {rpm_max_value:.2f}")

        x_axis_max = st.sidebar.number_input("X-axis maximum", value=rpm_max_value, min_value=1.0, max_value=100.0)

        df = df[(df[revolution_col] >= machine_params['n2']) & (df[revolution_col] <= machine_params['n1'])]

        def calculate_torque_wrapper(row):
            working_pressure = row[working_pressure_col]
            current_speed = row[revolution_col]

            if current_speed < machine_params['n1']:
                torque = working_pressure * machine_params['torque_constant']
            else:
                torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure

            return round(torque, 2)

        df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

        torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
        rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])

        df['Is_Anomaly'] = df[working_pressure_col] >= anomaly_threshold

        def M_max_Vg2(rpm):
            return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

        elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
        elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

        rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)

        fig, ax = plt.subplots(figsize=(14, 10))

        ax.plot(rpm_curve[rpm_curve <= elbow_rpm_cont],
                np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params['M_cont_value']),
                'g-', linewidth=2, label='M cont Max [kNm]')

        ax.plot(rpm_curve[rpm_curve <= elbow_rpm_max],
                np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']),
                'r-', linewidth=2, label='M max Vg1 [kNm]')

        ax.plot(rpm_curve[rpm_curve <= machine_params['n1']], M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]),
                'r--', linewidth=2, label='M max Vg2 [kNm]')

        ax.plot([elbow_rpm_max, elbow_rpm_max], [0, machine_params['M_max_Vg1']], color='purple', linestyle=':', linewidth=3)
        ax.plot([elbow_rpm_cont, elbow_rpm_cont], [0, machine_params['M_cont_value']], color='orange', linestyle=':', linewidth=3)
        ax.plot([machine_params['n1'], machine_params['n1']], [0, machine_params['M_cont_value']], color='black', linestyle='--', linewidth=2)

        normal_data = df[(~df['Is_Anomaly']) & (~df['Calculated torque [kNm]'].isin(torque_outliers))]
        anomaly_data = df[df['Is_Anomaly']]
        torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers) & (~df['Is_Anomaly'])]
        rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers) & (~df['Is_Anomaly'])]

        scatter_normal = ax.scatter(normal_data[revolution_col], normal_data['Calculated torque [kNm]'],
                                    c=normal_data['Calculated torque [kNm]'], cmap='viridis',
                                    s=50, alpha=0.6, label='Normal Data')
        scatter_anomaly = ax.scatter(anomaly_data[revolution_col], anomaly_data['Calculated torque [kNm]'],
                                     color='red', s=100, alpha=0.8, marker='X', label=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)')
        scatter_torque_outliers = ax.scatter(torque_outlier_data[revolution_col], torque_outlier_data['Calculated torque [kNm]'],
                                             color='orange', s=100, alpha=0.8, marker='D', label='Torque Outliers')
        scatter_rpm_outliers = ax.scatter(rpm_outlier_data[revolution_col], rpm_outlier_data['Calculated torque [kNm]'],
                                          color='purple', s=100, alpha=0.8, marker='s', label='RPM Outliers')

        ax.axhline(y=torque_upper_whisker, color='gray', linestyle='--', linewidth=1, label='Torque Upper Whisker')
        ax.axhline(y=torque_lower_whisker, color='gray', linestyle=':', linewidth=1, label='Torque Lower Whisker')

        ax.set_xlim(0, x_axis_max)
        ax.set_ylim(0, max(60, df['Calculated torque [kNm]'].max() * 1.1))
        ax.set_xlabel('Drehzahl / speed / revolutiones [1/min]')
        ax.set_ylabel('Drehmoment / torque / couple [kNm]')
        plt.title(f'{selected_machine} - Torque Analysis')

        ax.grid(True, which='both', linestyle=':', color='gray', alpha=0.5)

        ax.text(elbow_rpm_max * 0.5, machine_params['M_max_Vg1'] * 1.05, f'M max (max.): {machine_params["M_max_Vg1"]} kNm',
                fontsize=10, ha='center', va='bottom', color='red')
        ax.text(elbow_rpm_cont * 0.5, machine_params['M_cont_value'] * 0.95, f'M cont (max.): {machine_params["M_cont_value"]} kNm',
                fontsize=10, ha='center', va='top', color='green')

        ax.text(elbow_rpm_max, 0, f'{elbow_rpm_max:.2f}', ha='right', va='bottom', color='purple', fontsize=8)
        ax.text(elbow_rpm_cont, 0, f'{elbow_rpm_cont:.2f}', ha='right', va='bottom', color='orange', fontsize=8)
        ax.text(machine_params['n1'], machine_params['M_cont_value'], f'n1: {machine_params["n1"]}', ha='right', va='top', color='black', fontsize=8, rotation=90)

        cbar = plt.colorbar(scatter_normal)
        cbar.set_label('Calculated Torque [kNm]')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        st.sidebar.markdown("## Download Results")

        stats_df = pd.DataFrame({
            'RPM': rpm_stats,
            'Calculated Torque': df['Calculated torque [kNm]'].describe(),
            'Working Pressure': df[working_pressure_col].describe()
        })
        st.sidebar.markdown(get_table_download_link(stats_df, "statistical_analysis.csv", "Download Statistical Analysis"), unsafe_allow_html=True)

        plot_base64 = fig_to_base64(fig)
        href = f'<a href="data:image/png;base64,{plot_base64}" download="torque_analysis_plot.png">Download Plot</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

        result_analysis_df = pd.DataFrame({
            'Metric': ['Total data points', 'Normal data points', 'Anomaly data points', 'Percentage of anomalies',
                       'Elbow point Max', 'Elbow point Cont', 'Torque Upper Whisker', 'Torque Lower Whisker',
                       'Number of torque outliers', 'Percentage of torque outliers', 'RPM Upper Whisker', 'RPM Lower Whisker',
                       'Number of RPM outliers', 'Percentage of RPM outliers'],
            'Value': [len(df), len(normal_data), len(anomaly_data), f"{len(anomaly_data) / len(df) * 100:.2f}%",
                      f"{elbow_rpm_max:.2f}", f"{elbow_rpm_cont:.2f}", f"{torque_upper_whisker:.2f}",
                      f"{torque_lower_whisker:.2f}", len(torque_outliers),
                      f"{len(torque_outliers) / len(df) * 100:.2f}%", f"{rpm_upper_whisker:.2f}",
                      f"{rpm_lower_whisker:.2f}", len(rpm_outliers), f"{len(rpm_outliers) / len(df) * 100:.2f}%"]
        })
        st.sidebar.markdown(get_table_download_link(result_analysis_df, "result_analysis.csv", "Download Result Analysis"), unsafe_allow_html=True)

        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("RPM Statistics:")
            st.write(rpm_stats)

        with col2:
            st.write("Calculated Torque Statistics:")
            st.write(df['Calculated torque [kNm]'].describe())

        with col3:
            st.write("Working Pressure Statistics:")
            st.write(df[working_pressure_col].describe())

        st.subheader("Anomaly Detection Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"Total data points: {len(df)}")
            st.write(f"Normal data points: {len(normal_data)}")
            st.write(f"Anomaly data points: {len(anomaly_data)}")
            st.write(f"Percentage of anomalies: {len(anomaly_data) / len(df) * 100:.2f}%")

        with col2:
            st.write(f"Elbow point Max: {elbow_rpm_max:.2f} rpm")
            st.write(f"Elbow point Cont: {elbow_rpm_cont:.2f} rpm")

        with col3:
            st.write("Whisker and Outlier Information:")
            st.write(f"Torque Upper Whisker: {torque_upper_whisker:.2f} kNm")
            st.write(f"Torque Lower Whisker: {torque_lower_whisker:.2f} kNm")
            st.write(f"Number of torque outliers: {len(torque_outliers)}")
            st.write(f"Percentage of torque outliers: {len(torque_outliers) / len(df) * 100:.2f}%")
            st.write(f"RPM Upper Whisker: {rpm_upper_whisker:.2f} rpm")
            st.write(f"RPM Lower Whisker: {rpm_lower_whisker:.2f} rpm")
            st.write(f"Number of RPM outliers: {len(rpm_outliers)}")
            st.write(f"Percentage of RPM outliers: {len(rpm_outliers) / len(df) * 100:.2f}%")

        st.info("The Anomaly Detection Results highlight points in the data where the machine's behavior deviates from expected patterns. "
                "Anomalies are identified when the working pressure exceeds a defined threshold, which could indicate potential issues.")

    else:
        st.info("Please upload a Raw Data file to begin the analysis.")

    st.markdown("---")
    st.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")

if __name__ == "__main__":
    main()
