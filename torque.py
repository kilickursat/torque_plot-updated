import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

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


def load_machine_specs(file):
    """Load machine specifications from XLSX file."""
    specs_df = pd.read_excel(file)
    specs_df.columns = specs_df.columns.str.strip()  # Strip any leading/trailing whitespace and newlines
    
    # Display the columns in a more UX-friendly design
    with st.expander("Columns in the Uploaded Excel File"):
        st.dataframe(pd.DataFrame(specs_df.columns, columns=["Column Names"]))
    
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

def main():
    set_page_config()
    set_background_color()
    add_logo()

    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")

    # File uploaders
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")

    if machine_specs_file is None:
        st.warning("Please upload Machine Specifications XLSX file.")
        return

    try:
        machine_specs = load_machine_specs(machine_specs_file)
        machine_types = machine_specs['Projekt'].unique()
        selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)
        machine_params = get_machine_params(machine_specs, selected_machine)

        st.write("**Loaded Machine Parameters:**")
        st.dataframe(pd.DataFrame([machine_params]))
    except Exception as e:
        st.error(f"An error occurred while processing the machine specifications: {str(e)}")
        return

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    if raw_data_file is None:
        st.info("Please upload a Raw Data file to begin the analysis.")
        return

    df = load_data_file(raw_data_file)
    if df is None:
        st.error("Failed to load the file. Please check the format and try again.")
        return

    display_columns_with_hover(df)

    pressure_col = st.selectbox("Select pressure column", options=df.columns)
    revolution_col = st.selectbox("Select revolution column", options=df.columns)

    if not (pressure_col and revolution_col):
        st.warning("Please select both pressure and revolution columns to proceed with the analysis.")
        return

    # Data processing
    df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')
    df[pressure_col] = pd.to_numeric(df[pressure_col], errors='coerce')
    df = df.dropna(subset=[revolution_col, pressure_col])
    df = df[(df[revolution_col] >= machine_params['n2']) & (df[revolution_col] <= machine_params['n1'])]

    # Calculate torque
    df['Calculated torque [kNm]'] = df.apply(lambda row: calculate_torque(row, pressure_col, revolution_col, machine_params), axis=1)

    # Statistics
    rpm_stats = df[revolution_col].describe()
    rpm_max_value = rpm_stats['max']

    # Whiskers and outliers
    torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
    rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])

    # Anomaly detection
    df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold
    normal_data = df[(~df['Is_Anomaly']) & (~df['Calculated torque [kNm]'].isin(torque_outliers))]
    anomaly_data = df[df['Is_Anomaly']]

    # Calculate elbow points
    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

    # Plot
    fig = create_plot(df, machine_params, P_max, nu, anomaly_threshold, rpm_max_value, 
                      elbow_rpm_max, elbow_rpm_cont, torque_lower_whisker, torque_upper_whisker, 
                      normal_data, anomaly_data, torque_outliers, rpm_outliers, 
                      revolution_col, pressure_col)
    st.pyplot(fig)

    # Display statistics
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
        st.write(df[pressure_col].describe())

    # Anomaly Detection Results with Explanation
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

    # Short explanation for non-technical users
    st.info("The Anomaly Detection Results highlight points in the data where the machine's behavior deviates from expected patterns. "
            "Anomalies are identified when the working pressure exceeds a defined threshold, which could indicate potential issues. "
            "Outliers are data points that fall outside the normal range, which may also signal unusual conditions that warrant attention.")

    # Download buttons
    st.sidebar.markdown("## Download Results")
    
    # Statistical Analysis Results
    stats_df = pd.DataFrame({
        'RPM': rpm_stats,
        'Calculated Torque': df['Calculated torque [kNm]'].describe(),
        'Working Pressure': df[pressure_col].describe()
    })
    st.sidebar.markdown(get_table_download_link(stats_df, "statistical_analysis.csv", "Download Statistical Analysis"), unsafe_allow_html=True)

    # Plot
    plot_base64 = fig_to_base64(fig)
    href = f'<a href="data:image/png;base64,{plot_base64}" download="torque_analysis_plot.png">Download Plot</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

    # Result Analysis
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

    # Add footer with creator information
    st.markdown("---")
    st.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")
if __name__ == "__main__":
    main()
