import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import plotly.graph_objects as go
import plotly.express as px

# Update the sensor column map with more potential column names
sensor_column_map = {
    "pressure": ["Working pressure [bar]", "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26", "Pression [bar]", "Presión [bar]", "Pressure", "Pressure [bar]", "Working Pressure"],
    "revolution": ["Revolution [rpm]", "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30", "Vitesse [rpm]", "Revoluciones [rpm]", "RPM", "Speed", "Rotation Speed"]
}

def find_sensor_columns(df):
    found_columns = {}
    for sensor, possible_names in sensor_column_map.items():
        for name in possible_names:
            if name in df.columns:
                found_columns[sensor] = name
                break
        if sensor not in found_columns:
            # If exact match not found, try case-insensitive partial match
            for col in df.columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    found_columns[sensor] = col
                    break
    return found_columns


@st.cache_data
def load_machine_specs(file):
    """Load machine specifications from XLSX file."""
    specs_df = pd.read_excel(file)
    specs_df.columns = specs_df.columns.str.strip()
    return specs_df

def get_machine_params(specs_df, machine_type):
    """Extract relevant machine parameters based on machine type."""
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
    """Calculate whiskers and outliers for a given dataset."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    whisker_length = 1.5 * IQR
    lower_whisker = Q1 - whisker_length
    upper_whisker = Q3 + whisker_length
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    return lower_whisker, upper_whisker, outliers

def set_page_config():
    """Set the Streamlit page configuration."""
    st.set_page_config(
        page_title="Herrenknecht Torque Analysis",
        page_icon="https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png",
        layout="wide"
    )

def set_background_color():
    """Set the background color for the app."""
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
    """Add a sidebar logo for the app."""
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
            height: 100px; /* Reduced height */
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0rem;
        }
        .sidebar-content {
            padding-top: 100px; /* Same as the ::before height */
        }
        .sidebar-content > * {
            margin-bottom: 0.5rem !important; /* Reduce space between sidebar elements */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_table_download_link(df, filename, text):
    """Generate a download link for a pandas DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    return img_str

@st.cache_data
def load_data_file(file):
    """Loads a CSV or XLSX file and returns a pandas DataFrame."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=';', decimal=',')
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        return None
    return df

def display_columns_with_hover(df):
    """Displays the column names with hover-over tooltips for each column."""
    with st.expander("View and Hover Over Columns"):
        st.write("Hover over the column names to inspect them:")
        
        # Create a DataFrame to show the column names with hover
        for col in df.columns:
            st.write(f"**{col}**", help=f"Column '{col}' has {df[col].notna().sum()} non-null values.")

def get_column(df, sensor_name):
    """Find the correct column in the DataFrame based on the sensor name mapping."""
    possible_columns = sensor_column_map[sensor_name]
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

@st.cache_data
def load_machine_specs(file):
    """Load machine specifications from XLSX file."""
    specs_df = pd.read_excel(file)
    specs_df.columns = specs_df.columns.str.strip()
    return specs_df

@st.cache_data
def load_data_file(file):
    """Loads a CSV or XLSX file and returns a pandas DataFrame."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=';', decimal=',')
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        return None
    return df

def set_page_config():
    st.set_page_config(
        page_title="Herrenknecht Torque Analysis",
        page_icon="https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png",
        layout="wide"
    )

def main():
    set_page_config()
    
    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")
    st.markdown("Created by Kursat KILIC - Geotechnical Digitalization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")
    
    with col2:
        raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])

    if machine_specs_file is not None and raw_data_file is not None:
        machine_specs = load_machine_specs(machine_specs_file)
        df = load_data_file(raw_data_file)

        if machine_specs is not None and df is not None:
            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.selectbox("Select Machine Type", machine_types)
            
            machine_params = get_machine_params(machine_specs, selected_machine)
            
            st.markdown(
                f"""
                <style>
                .dataframe {{
                    background-color: rgb(0, 62, 37);
                    color: white;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            st.dataframe(pd.DataFrame([machine_params]))

            pressure_col = st.selectbox("Select pressure column", options=df.columns)
            revolution_col = st.selectbox("Select revolution column", options=df.columns)

            if pressure_col and revolution_col:
                df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')
                df[pressure_col] = pd.to_numeric(df[pressure_col], errors='coerce')
                df = df.dropna(subset=[revolution_col, pressure_col])

                # Calculate torque
                df['Calculated torque [kNm]'] = df.apply(lambda row: calculate_torque(row, pressure_col, revolution_col, machine_params), axis=1)

                # Anomaly detection and outlier calculation
                anomaly_threshold = st.slider("Anomaly threshold (bar)", min_value=100, max_value=500, value=250)
                df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold
                torque_lower_whisker, torque_upper_whisker = np.percentile(df['Calculated torque [kNm]'], [25, 75])
                rpm_lower_whisker, rpm_upper_whisker = np.percentile(df[revolution_col], [25, 75])

                # Create Plotly figure
                fig = make_subplots(rows=1, cols=1, subplot_titles=[f'{selected_machine} - Torque Analysis'])

                # Plot torque curves and data points
                plot_torque_curves(fig, machine_params, df, revolution_col, pressure_col, anomaly_threshold, torque_lower_whisker, torque_upper_whisker, rpm_lower_whisker, rpm_upper_whisker)

                # Update layout for better visibility
                fig.update_layout(
                    height=800,  # Increase the height of the plot
                    xaxis_title='Revolution [1/min]',
                    yaxis_title='Torque [kNm]',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )

                st.plotly_chart(fig, use_container_width=True)

                display_statistics(df, revolution_col, pressure_col)
                display_explanation(anomaly_threshold)

    else:
        st.info("Please upload both Machine Specifications XLSX and Raw Data files to begin the analysis.")

@st.cache_data
def get_machine_params(specs_df, machine_type):
    """Extract relevant machine parameters based on machine type."""
    machine_data = specs_df[specs_df['Projekt'] == machine_type].iloc[0]
    
    param_mapping = {
        'n1': ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]'],
        'n2': ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]'],
        'M_cont_value': ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)'],
        'M_max_Vg1': ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]'],
        'torque_constant': ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]']
    }

    params = {}
    for param, possible_names in param_mapping.items():
        for name in possible_names:
            if name in machine_data.index:
                params[param] = machine_data[name]
                break
    
    return params

def calculate_torque(row, pressure_col, revolution_col, machine_params):
    working_pressure = row[pressure_col]
    current_speed = row[revolution_col]

    if current_speed < machine_params['n1']:
        torque = working_pressure * machine_params['torque_constant']
    else:
        torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure

    return round(torque, 2)

def plot_torque_curves(fig, machine_params, df, revolution_col, pressure_col, anomaly_threshold, torque_lower_whisker, torque_upper_whisker, rpm_lower_whisker, rpm_upper_whisker):
    P_max = 132.0  # You may want to make this a parameter
    nu = 0.7  # You may want to make this a parameter

    rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)
    
    def M_max_Vg2(rpm):
        return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

    # Plot torque curves
    fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_cont], y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params['M_cont_value']), mode='lines', name='M cont Max [kNm]', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_max], y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']), mode='lines', name='M max Vg1 [kNm]', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= machine_params['n1']], y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]), mode='lines', name='M max Vg2 [kNm]', line=dict(color='red', width=2, dash='dash')))

    # Add vertical lines at elbow points
    fig.add_vline(x=elbow_rpm_max, line_dash="dot", line_color="purple")
    fig.add_vline(x=elbow_rpm_cont, line_dash="dot", line_color="orange")
    fig.add_vline(x=machine_params['n1'], line_dash="dash", line_color="black")

    # Separate normal and anomaly data
    normal_data = df[~df['Is_Anomaly']]
    anomaly_data = df[df['Is_Anomaly']]

    # Plot data points
    fig.add_trace(go.Scatter(x=normal_data[revolution_col], y=normal_data['Calculated torque [kNm]'], mode='markers', name='Normal Data', marker=dict(color=normal_data['Calculated torque [kNm]'], colorscale='Viridis', size=8)))
    fig.add_trace(go.Scatter(x=anomaly_data[revolution_col], y=anomaly_data['Calculated torque [kNm]'], mode='markers', name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)', marker=dict(color='red', symbol='x', size=10)))

    # Add horizontal lines for the torque whiskers
    fig.add_hline(y=torque_upper_whisker, line_dash="dash", line_color="gray", annotation_text="Torque Upper Whisker")
    fig.add_hline(y=torque_lower_whisker, line_dash="dot", line_color="gray", annotation_text="Torque Lower Whisker")

def display_statistics(df, revolution_col, pressure_col):
    st.subheader("Statistical Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("RPM Statistics:")
        st.write(df[revolution_col].describe())

    with col2:
        st.write("Calculated Torque Statistics:")
        st.write(df['Calculated torque [kNm]'].describe())

    with col3:
        st.write("Working Pressure Statistics:")
        st.write(df[pressure_col].describe())

def display_explanation(anomaly_threshold):
    st.subheader("Understanding the Results")
    st.write(f"""
    This analysis provides insights into the performance of the machine:

    1. **Normal Data**: These are the typical operating points of the machine. They represent the standard working conditions and fall within expected ranges for revolution, torque, and pressure.

    2. **Anomalies**: These are instances where the working pressure exceeds the set threshold (currently set to {anomaly_threshold} bar). Anomalies might indicate:
       - Unusual operating conditions
       - Potential issues with the machine
       - Extreme workloads

    3. **Outliers**: These are data points that fall significantly outside the normal range for either torque or RPM. Outliers may represent:
       - Extreme operating conditions
       - Measurement errors
       - Temporary spikes in performance

    The statistical summary shows:
    - **Mean**: The average value, giving you a sense of the typical operating point.
    - **Median (50%)**: The middle value when data is sorted, useful for understanding the central tendency without being affected by extreme values.
    - **Standard Deviation (std)**: Measures the spread of the data. A larger standard deviation indicates more variability in the measurements.
    - **Min and Max**: The lowest and highest values recorded, helping to understand the range of operation.
    - **25%, 50%, 75% (Quartiles)**: These split the data into four equal parts, giving you an idea of the data's distribution.

    Understanding these statistics can help identify:
    - Typical operating ranges
    - Unusual patterns in machine operation
    - Potential areas for optimization or maintenance

    If you notice a high number of anomalies or outliers, or if the statistics show unexpected values, it may be worth investigating further or consulting with a technical expert for a more detailed analysis.
    """)

if __name__ == "__main__":
    main()
