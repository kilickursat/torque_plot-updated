import streamlit as st
import numpy as np
import pandas as pd
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta



# Optimization: Add cache decorator to improve performance on repeated file loads
@st.cache_data
def load_data(file, file_type):
    try:
        if file_type == 'csv':
            # Try different separators and encodings
            for sep in [';', ',']:
                for encoding in ['utf-8', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file, sep=sep, encoding=encoding, decimal=',')
                        return df
                    except:
                        pass
            raise ValueError("Unable to read CSV file with tried separators and encodings")
        elif file_type == 'xlsx':
            df = pd.read_excel(file)
            return df
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Update the sensor column map with more potential column names
sensor_column_map = {
    "pressure": ["Working pressure [bar]", "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26", "Pression [bar]", "Presión [bar]", "Pressure", "Pressure [bar]", "Working Pressure"],
    "revolution": ["Revolution [rpm]", "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30", "Vitesse [rpm]", "Revoluciones [rpm]", "RPM", "Speed", "Rotation Speed"],
    "time": ["Time", "Timestamp", "DateTime", "Date", "Zeit", "Relativzeit", "Uhrzeit", "Datum"],
    "advance_rate": ["Advance Rate", "Vorschubgeschwindigkeit", "Avance", "Rate of Penetration", "ROP", "Advance [m/min]", "Advance [mm/min]"],
    "thrust_force": ["Thrust Force", "Thrust", "Vorschubkraft", "Force", "Force at Cutting Head", "Thrust Force [kN]"]
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
def load_machine_specs(file, file_type):
    """Load machine specifications from XLSX or CSV file."""
    try:
        if file_type == 'xlsx':
            specs_df = pd.read_excel(file)
        elif file_type == 'csv':
            specs_df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file type")
        specs_df.columns = specs_df.columns.str.strip()  # Strip any leading/trailing whitespace and newlines
        return specs_df
    except Exception as e:
        st.error(f"Error loading machine specifications: {str(e)}")
        return None

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

def calculate_whisker_and_outliers_advanced(data):
    """Calculate whiskers and outliers using 10th and 90th percentiles."""
    Q1 = data.quantile(0.10)
    Q3 = data.quantile(0.90)
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
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_table_download_link(df, filename, text):
    """Generate a download link for a pandas DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_statistics(df, revolution_col, pressure_col, thrust_force_col=None):
    """Display statistics of RPM, Torque, Pressure, and Thrust Force per Cutting Ring."""
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

    if thrust_force_col is not None and 'Thrust Force per Cutting Ring' in df.columns:
        st.write("**Thrust Force per Cutting Ring Statistics:**")
        st.write(df['Thrust Force per Cutting Ring'].describe())

def display_explanation(anomaly_threshold):
    """Display an explanation of the results."""
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

def main():
    set_page_config()
    set_background_color()
    add_logo()

    # Page selection
    page = st.sidebar.selectbox("Select Page", ("Original Analysis", "Advanced Analysis"))

    if page == "Original Analysis":
        original_page()
    elif page == "Advanced Analysis":
        advanced_page()

def original_page():
    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")

    # File uploaders for batch data
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")

    # Load machine specs if available
    if machine_specs_file is not None:
        try:
            machine_specs = load_machine_specs(machine_specs_file, 'xlsx')
            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

            machine_params = get_machine_params(machine_specs, selected_machine)

            # Convert the machine parameters to a DataFrame
            params_df = pd.DataFrame([machine_params])
            # Create a styled HTML table with thicker borders
            styled_table = params_df.style.set_table_styles([
                {'selector': 'th', 'props': [('border', '2px solid black'), ('padding', '5px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
                {'selector': '', 'props': [('border-collapse', 'collapse')]}
            ]).to_html()

            # Remove the unwanted CSS that appears above the table
            styled_table = styled_table.split('</style>')[-1]

            # Display the styled table
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
                table th,
                table td {{
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
            st.stop()
    else:
        st.warning("Please upload Machine Specifications XLSX file.")
        return

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    if raw_data_file is not None:
        # Load data
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            # Find sensor columns
            sensor_columns = find_sensor_columns(df)

            # Allow user to select columns
            pressure_col = st.selectbox("Select pressure column", options=df.columns, index=df.columns.get_loc(sensor_columns.get('pressure', df.columns[0])))
            revolution_col = st.selectbox("Select revolution column", options=df.columns, index=df.columns.get_loc(sensor_columns.get('revolution', df.columns[0])))

            if pressure_col and revolution_col:
                # Proceed with data processing and visualization
                df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')
                df[pressure_col] = pd.to_numeric(df[pressure_col], errors='coerce')
                df = df.dropna(subset=[revolution_col, pressure_col])

                # RPM Statistics
                rpm_stats = df[revolution_col].describe()
                rpm_max_value = rpm_stats['max']
                st.sidebar.write(f"Recommended value for x-axis based on the Max RPM in Data: {rpm_max_value:.2f}")

                # Allow user to set x_axis_max
                x_axis_max = st.sidebar.number_input("X-axis maximum", value=float(rpm_max_value), min_value=1.0, max_value=float(rpm_max_value * 1.2))

                # Filter data points between n2 and n1 rpm
                df = df[(df[revolution_col] >= machine_params['n2']) & (df[revolution_col] <= machine_params['n1'])]

                # Calculate torque
                def calculate_torque_wrapper(row):
                    working_pressure = row[pressure_col]
                    current_speed = row[revolution_col]

                    if current_speed < machine_params['n1']:
                        torque = working_pressure * machine_params['torque_constant']
                    else:
                        torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure

                    return round(torque, 2)

                df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

                # Calculate whiskers and outliers for torque
                torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
                rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])

                # Anomaly detection based on working pressure
                df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold

                # Function to calculate M max Vg2
                def M_max_Vg2(rpm):
                    return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

                # Calculate the elbow points for the max and continuous torque
                elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
                elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

                # Generate RPM values for the torque curve
                rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)  # Avoid division by zero

                fig = make_subplots(rows=1, cols=1)

                # Plot torque curves
                fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_cont],
                                         y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params['M_cont_value']),
                                         mode='lines', name='M cont Max [kNm]', line=dict(color='green', width=2)))

                fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= elbow_rpm_max],
                                         y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']),
                                         mode='lines', name='M max Vg1 [kNm]', line=dict(color='red', width=2)))

                fig.add_trace(go.Scatter(x=rpm_curve[rpm_curve <= machine_params['n1']],
                                         y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]),
                                         mode='lines', name='M max Vg2 [kNm]', line=dict(color='red', width=2, dash='dash')))

                # Calculate the y-values for the vertical lines
                y_max_vg2 = M_max_Vg2(np.array([elbow_rpm_max, elbow_rpm_cont, machine_params['n1']]))

                # Add truncated vertical lines at elbow points
                fig.add_trace(go.Scatter(x=[elbow_rpm_max, elbow_rpm_max], y=[0, y_max_vg2[0]],
                                         mode='lines', line=dict(color='purple', width=1, dash='dot'), showlegend=False))
                fig.add_trace(go.Scatter(x=[elbow_rpm_cont, elbow_rpm_cont], y=[0, y_max_vg2[1]],
                                         mode='lines', line=dict(color='orange', width=1, dash='dot'), showlegend=False))
                fig.add_trace(go.Scatter(x=[machine_params['n1'], machine_params['n1']], y=[0, y_max_vg2[2]],
                                         mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False))

                # Separate normal and anomaly data
                normal_data = df[~df['Is_Anomaly']]
                anomaly_data = df[df['Is_Anomaly']]

                # Separate outlier data
                torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers)]
                rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

                # Plot data points
                fig.add_trace(go.Scatter(x=normal_data[revolution_col], y=normal_data['Calculated torque [kNm]'],
                                         mode='markers', name='Normal Data',
                                         marker=dict(color=normal_data['Calculated torque [kNm]'], colorscale='Viridis', size=8)))

                fig.add_trace(go.Scatter(x=anomaly_data[revolution_col], y=anomaly_data['Calculated torque [kNm]'],
                                         mode='markers', name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)',
                                         marker=dict(color='red', symbol='x', size=10)))

                fig.add_trace(go.Scatter(x=torque_outlier_data[revolution_col], y=torque_outlier_data['Calculated torque [kNm]'],
                                         mode='markers', name='Torque Outliers',
                                         marker=dict(color='orange', symbol='diamond', size=10)))

                fig.add_trace(go.Scatter(x=rpm_outlier_data[revolution_col], y=rpm_outlier_data['Calculated torque [kNm]'],
                                         mode='markers', name='RPM Outliers',
                                         marker=dict(color='purple', symbol='square', size=10)))

                # Add horizontal lines for the torque whiskers
                fig.add_hline(y=torque_upper_whisker, line_dash="dash", line_color="gray", annotation_text="Torque Upper Whisker")
                fig.add_hline(y=torque_lower_whisker, line_dash="dot", line_color="gray", annotation_text="Torque Lower Whisker")

                # Set plot layout with adjusted dimensions
                fig.update_layout(
                    title=f'{selected_machine} - Torque Analysis',
                    xaxis_title='Revolution [1/min]',
                    yaxis_title='Torque [kNm]',
                    xaxis=dict(range=[0, x_axis_max]),
                    yaxis=dict(range=[0, max(60, df['Calculated torque [kNm]'].max() * 1.1)]),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    width=1000,
                    height=800,
                    margin=dict(l=50, r=50, t=100, b=100)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display the statistical summary
                display_statistics(df, revolution_col, pressure_col)

                # Provide an explanation of the analysis
                display_explanation(anomaly_threshold)

                # Download buttons for analysis results
                st.sidebar.markdown("## Download Results")
                stats_df = pd.DataFrame({
                    'RPM': df[revolution_col].describe(),
                    'Calculated Torque': df['Calculated torque [kNm]'].describe(),
                    'Working Pressure': df[pressure_col].describe()
                })
                st.sidebar.markdown(get_table_download_link(stats_df, "statistical_analysis.csv", "Download Statistical Analysis"), unsafe_allow_html=True)

    else:
        st.info("Please upload a Raw Data file to begin the analysis.")


# Helper Function Defined Outside
def format_timedelta(td):
    """
    Convert a pandas Timedelta to a human-readable string.
    
    Parameters:
        td (pd.Timedelta): The timedelta object to format.
        
    Returns:
        str: A formatted string representing the timedelta.
    """
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)      # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)        # 60 seconds in a minute
    fractional_seconds = td.microseconds / 1_000_000  # Convert microseconds to seconds

    formatted_time = ""
    if days > 0:
        formatted_time += f"{days} day{'s' if days != 1 else ''}, "
    if hours > 0:
        formatted_time += f"{hours} hour{'s' if hours != 1 else ''}, "
    if minutes > 0:
        formatted_time += f"{minutes} minute{'s' if minutes != 1 else ''}, "

    # Combine seconds and fractional seconds
    total_sec = seconds + fractional_seconds
    formatted_time += f"{total_sec:.2f} second{'s' if total_sec != 1 else ''}"

    return formatted_time

# Main Function
def advanced_page():
    st.title("Advanced Analysis")

    # File uploaders for batch data
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader(
        "Upload Machine Specifications: XLSX (MM-Baureihenliste) or CSV format accepted",
        type=["xlsx", "csv"],
    )

    # Load machine specs if available
    if machine_specs_file is not None:
        try:
            file_type = machine_specs_file.name.split(".")[-1].lower()
            machine_specs = load_machine_specs(machine_specs_file, file_type)
            if machine_specs is None or machine_specs.empty:
                st.error("Machine specifications file is empty or could not be loaded.")
                st.stop()

            if "Projekt" not in machine_specs.columns:
                st.error("The machine specifications file must contain a 'Projekt' column.")
                st.stop()

            machine_types = machine_specs["Projekt"].unique()
            if len(machine_types) == 0:
                st.error("No machine types found in the specifications file.")
                st.stop()

            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved.")
                st.stop()

            # Display machine parameters
            params_df = pd.DataFrame([machine_params])
            styled_table = params_df.style.set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [("border", "2px solid black"), ("padding", "5px")],
                    },
                    {
                        "selector": "td",
                        "props": [("border", "2px solid black"), ("padding", "5px")],
                    },
                    {"selector": "", "props": [("border-collapse", "collapse")]},
                ]
            ).to_html()

            styled_table = styled_table.split("</style>")[-1]

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
                table th,
                table td {{
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
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(
                f"An error occurred while processing the machine specifications: {str(e)}"
            )
            st.stop()
    else:
        st.warning("Please upload Machine Specifications file.")
        return

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input(
        "Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0
    )
    nu = st.sidebar.number_input(
        "Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0
    )
    anomaly_threshold = st.sidebar.number_input(
        "Anomaly threshold (bar)", value=250, min_value=100, max_value=500
    )
    num_cutting_rings = st.sidebar.number_input(
        "Number of Cutting Rings", value=1, min_value=1, max_value=100
    )

    if raw_data_file is not None:
        # Load data
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            # Find sensor columns
            sensor_columns = find_sensor_columns(df)

            # Allow user to select columns if not found or adjust selections
            st.subheader("Select Sensor Columns")
            # Time Column
            if "time" in sensor_columns and sensor_columns["time"] in df.columns:
                default_time_col = sensor_columns["time"]
            else:
                default_time_col = df.columns[0]
            time_col = st.selectbox(
                "Select Time Column",
                options=df.columns,
                index=df.columns.get_loc(default_time_col),
            )

            # Pressure Column
            if "pressure" in sensor_columns and sensor_columns["pressure"] in df.columns:
                default_pressure_col = sensor_columns["pressure"]
            else:
                default_pressure_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            pressure_col = st.selectbox(
                "Select Pressure Column",
                options=df.columns,
                index=df.columns.get_loc(default_pressure_col) if default_pressure_col else 0,
            )

            # Revolution Column
            if "revolution" in sensor_columns and sensor_columns["revolution"] in df.columns:
                default_revolution_col = sensor_columns["revolution"]
            else:
                default_revolution_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
            revolution_col = st.selectbox(
                "Select Revolution Column",
                options=df.columns,
                index=df.columns.get_loc(default_revolution_col) if default_revolution_col else 0,
            )

            # Advance Rate Column
            if "advance_rate" in sensor_columns and sensor_columns["advance_rate"] in df.columns:
                default_advance_rate_col = sensor_columns["advance_rate"]
            else:
                default_advance_rate_col = df.columns[3] if len(df.columns) > 3 else df.columns[0]
            advance_rate_col = st.selectbox(
                "Select Advance Rate Column",
                options=df.columns,
                index=df.columns.get_loc(default_advance_rate_col) if default_advance_rate_col else 0,
            )

            # Thrust Force Column
            if "thrust_force" in sensor_columns and sensor_columns["thrust_force"] in df.columns:
                default_thrust_force_col = sensor_columns["thrust_force"]
            else:
                default_thrust_force_col = df.columns[4] if len(df.columns) > 4 else df.columns[0]
            thrust_force_col = st.selectbox(
                "Select Thrust Force Column",
                options=df.columns,
                index=df.columns.get_loc(default_thrust_force_col) if default_thrust_force_col else 0,
            )

            # Ensure time column is appropriately parsed
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
            if df[time_col].isnull().all():
                st.error(
                    f"The selected time column '{time_col}' cannot be converted to numeric values."
                )
                return

            # Ask the user to select the unit of the time column
            time_unit = st.selectbox(
                "Select Time Unit for Time Column",
                options=["milliseconds", "seconds", "minutes", "hours"],
                index=0,
                help="Choose the unit that matches the time data in your dataset."
            )

            # Display the maximum value in the time column for debugging
            max_time_value = df[time_col].max()
            st.write(f"**Maximum value in the time column (`{time_col}`):** {max_time_value} {time_unit}")

            # Define the maximum allowed value based on the selected time unit
            max_allowed_value = {
                "milliseconds": 2**63 // 1_000_000,
                "seconds": 2**63 // 1_000_000_000,
                "minutes": 2**63 // (60 * 1_000_000_000),
                "hours": 2**63 // (3600 * 1_000_000_000),
            }[time_unit]

            # Display the maximum allowed value for debugging
            st.write(f"**Maximum allowed value for '{time_unit}':** {max_allowed_value} {time_unit}")

            # Convert time column to seconds based on the selected unit
            if time_unit == "milliseconds":
                df["Time_unit"] = df[time_col] / 1000  # Convert to seconds
            elif time_unit == "seconds":
                df["Time_unit"] = df[time_col]
            elif time_unit == "minutes":
                df["Time_unit"] = df[time_col] * 60  # Convert to seconds
            elif time_unit == "hours":
                df["Time_unit"] = df[time_col] * 3600  # Convert to seconds

            # Check for out-of-bounds values after conversion
            if df["Time_unit"].max() > max_allowed_value:
                st.error(
                    f"The values in the time column exceed the maximum allowed for the selected unit '{time_unit}'. Please check the data or select a different unit."
                )
                return

            # Sort the dataframe by Time_unit
            df = df.sort_values("Time_unit")

            # Calculate min and max time
            min_time_unit = df["Time_unit"].min()
            max_time_unit = df["Time_unit"].max()

            # Display the time range in numeric format (seconds)
            st.write(f"**Data time range:** {min_time_unit:.2f} seconds to {max_time_unit:.2f} seconds")

            # Convert numeric Time_unit back to timedelta for human-readable format
            try:
                df["Human_Readable_Time"] = pd.to_timedelta(df["Time_unit"], unit='s')
            except Exception as e:
                st.error(f"Error converting Time_unit to timedelta: {e}")
                return

            # Format the min and max times using the external helper function
            min_hr_time = format_timedelta(df["Human_Readable_Time"].min())
            max_hr_time = format_timedelta(df["Human_Readable_Time"].max())
            st.write(f"**Data time range (Human Readable):** {min_hr_time} to {max_hr_time}")

            # Define the format for the slider based on the time unit
            slider_format = "%.2f"

            # Create the time range slider
            time_range = st.slider(
                "Select Time Range",
                min_value=float(min_time_unit),
                max_value=float(max_time_unit),
                value=(float(min_time_unit), float(max_time_unit)),
                format=slider_format,
            )

            # Filter data based on the selected time range
            df = df[(df["Time_unit"] >= time_range[0]) & (df["Time_unit"] <= time_range[1])]

            # Ensure numeric columns are numeric
            for col in [
                pressure_col,
                revolution_col,
                advance_rate_col,
                thrust_force_col,
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop rows with NaNs in these columns
            df = df.dropna(
                subset=[pressure_col, revolution_col, advance_rate_col, thrust_force_col]
            )

            # Remove rows where revolution is zero to avoid division by zero
            df = df[df[revolution_col] != 0]

            # Calculate Penetration Rate as Advance Rate divided by Revolution
            df["Calculated Penetration Rate"] = (
                df[advance_rate_col] / df[revolution_col]
            )

            # Calculate Thrust Force per Cutting Ring
            df["Thrust Force per Cutting Ring"] = df[thrust_force_col] / num_cutting_rings

            # RPM Statistics
            rpm_stats = df[revolution_col].describe()
            rpm_max_value = rpm_stats["max"]
            st.sidebar.write(
                f"Recommended value for x-axis based on the Max RPM in Data: {rpm_max_value:.2f}"
            )

            # Allow user to set x_axis_max
            x_axis_max = st.sidebar.number_input(
                "X-axis maximum",
                value=float(rpm_max_value),
                min_value=1.0,
                max_value=float(rpm_max_value * 1.2),
            )

            # Filter data points between n2 and n1 rpm
            n2 = machine_params.get("n2", df[revolution_col].min())
            n1 = machine_params.get("n1", df[revolution_col].max())
            df = df[
                (df[revolution_col] >= n2)
                & (df[revolution_col] <= n1)
            ]

            # Calculate torque
            def calculate_torque_wrapper(row):
                working_pressure = row[pressure_col]
                current_speed = row[revolution_col]

                if current_speed < machine_params["n1"]:
                    torque = working_pressure * machine_params["torque_constant"]
                else:
                    torque = (
                        (machine_params["n1"] / current_speed)
                        * machine_params["torque_constant"]
                        * working_pressure
                    )

                return round(torque, 2)

            df["Calculated torque [kNm]"] = df.apply(
                calculate_torque_wrapper, axis=1
            )

            # Calculate whiskers and outliers using 10th and 90th percentiles
            (
                torque_lower_whisker,
                torque_upper_whisker,
                torque_outliers,
            ) = calculate_whisker_and_outliers_advanced(df["Calculated torque [kNm]"])
            (
                rpm_lower_whisker,
                rpm_upper_whisker,
                rpm_outliers,
            ) = calculate_whisker_and_outliers_advanced(df[revolution_col])

            # Anomaly detection based on working pressure
            df["Is_Anomaly"] = df[pressure_col] >= anomaly_threshold

            # Function to calculate M max Vg2
            def M_max_Vg2(rpm):
                return np.minimum(
                    machine_params["M_max_Vg1"],
                    (P_max * 60 * nu) / (2 * np.pi * rpm),
                )

            # Calculate the elbow points for the max and continuous torque
            elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params["M_max_Vg1"])
            elbow_rpm_cont = (
                P_max * 60 * nu
            ) / (2 * np.pi * machine_params["M_cont_value"])

            # Generate RPM values for the torque curve
            rpm_curve = np.linspace(0.1, machine_params["n1"], 1000)  # Avoid division by zero

            fig = make_subplots(rows=1, cols=1)

            # Plot torque curves
            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= elbow_rpm_cont],
                    y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params["M_cont_value"]),
                    mode="lines",
                    name="M cont Max [kNm]",
                    line=dict(color="green", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= elbow_rpm_max],
                    y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params["M_max_Vg1"]),
                    mode="lines",
                    name="M max Vg1 [kNm]",
                    line=dict(color="red", width=2),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= machine_params["n1"]],
                    y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params["n1"]]),
                    mode="lines",
                    name="M max Vg2 [kNm]",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            # Calculate the y-values for the vertical lines
            y_max_vg2 = M_max_Vg2(
                np.array(
                    [
                        elbow_rpm_max,
                        elbow_rpm_cont,
                        machine_params["n1"],
                    ]
                )
            )

            # Add truncated vertical lines at elbow points
            fig.add_trace(
                go.Scatter(
                    x=[elbow_rpm_max, elbow_rpm_max],
                    y=[0, y_max_vg2[0]],
                    mode="lines",
                    line=dict(color="purple", width=1, dash="dot"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[elbow_rpm_cont, elbow_rpm_cont],
                    y=[0, y_max_vg2[1]],
                    mode="lines",
                    line=dict(color="orange", width=1, dash="dot"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[machine_params["n1"], machine_params["n1"]],
                    y=[0, y_max_vg2[2]],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    showlegend=False,
                )
            )

            # Separate normal and anomaly data
            normal_data = df[~df["Is_Anomaly"]]
            anomaly_data = df[df["Is_Anomaly"]]

            # Separate outlier data
            torque_outlier_data = df[df["Calculated torque [kNm]"].isin(torque_outliers)]
            rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

            # Plot data points
            fig.add_trace(
                go.Scatter(
                    x=normal_data[revolution_col],
                    y=normal_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="Normal Data",
                    marker=dict(
                        color=normal_data["Calculated torque [kNm]"],
                        colorscale="Viridis",
                        size=8,
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=anomaly_data[revolution_col],
                    y=anomaly_data["Calculated torque [kNm]"],
                    mode="markers",
                    name=f"Anomaly (Pressure ≥ {anomaly_threshold} bar)",
                    marker=dict(color="red", symbol="x", size=10),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=torque_outlier_data[revolution_col],
                    y=torque_outlier_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="Torque Outliers",
                    marker=dict(color="orange", symbol="diamond", size=10),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=rpm_outlier_data[revolution_col],
                    y=rpm_outlier_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="RPM Outliers",
                    marker=dict(color="purple", symbol="square", size=10),
                )
            )

            # Add horizontal lines for the torque whiskers
            fig.add_hline(
                y=torque_upper_whisker,
                line_dash="dash",
                line_color="gray",
                annotation_text="Torque Upper Whisker (90th Percentile)",
            )
            fig.add_hline(
                y=torque_lower_whisker,
                line_dash="dot",
                line_color="gray",
                annotation_text="Torque Lower Whisker (10th Percentile)",
            )

            # Set plot layout with adjusted dimensions
            fig.update_layout(
                title=f"{selected_machine} - Advanced Torque Analysis",
                xaxis_title="Revolution [1/min]",
                yaxis_title="Torque [kNm]",
                xaxis=dict(range=[0, x_axis_max]),
                yaxis=dict(
                    range=[
                        0,
                        max(60, df["Calculated torque [kNm]"].max() * 1.1),
                    ]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                ),
                width=1000,
                height=800,
                margin=dict(l=50, r=50, t=100, b=100),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display the statistical summary
            display_statistics(df, revolution_col, pressure_col, thrust_force_col)

            # Additional Statistical Features
            st.subheader("Additional Statistical Features")

            # Advance Rate
            st.write("**Advance Rate Statistics:**")
            st.write(df[advance_rate_col].describe())

            # Calculated Penetration Rate
            st.write("**Penetration Rate Statistics (Calculated):**")
            st.write(df["Calculated Penetration Rate"].describe())

            # Thrust Force at the Cutting Head
            st.write("**Thrust Force at the Cutting Head Statistics:**")
            st.write(df[thrust_force_col].describe())

            # Thrust Force per Cutting Ring
            st.write("**Thrust Force per Cutting Ring Statistics:**")
            st.write(df["Thrust Force per Cutting Ring"].describe())

            # Plot features over Time as separate subplots
            st.subheader("Features over Time")

            # Define the features with their display names and colors
            features = [
                {"column": advance_rate_col, "display_name": "Advance Rate", "color": "blue"},
                {"column": "Calculated Penetration Rate", "display_name": "Penetration Rate", "color": "green"},
                {"column": thrust_force_col, "display_name": "Thrust Force", "color": "red"},
                {"column": "Thrust Force per Cutting Ring", "display_name": "Thrust Force per Cutting Ring", "color": "orange"},
                {"column": revolution_col, "display_name": "Revolution", "color": "purple"},
                {"column": pressure_col, "display_name": "Working Pressure", "color": "cyan"},
            ]

            num_features = len(features)

            # Optional: Allow users to set rolling window size
            window_size = st.sidebar.slider(
                "Select Rolling Window Size for Mean Calculation",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Adjust the window size to smooth the data. A larger window provides a smoother mean."
            )

            # Optional: Allow users to toggle mean lines
            show_means = st.checkbox("Show Mean Values", value=True, help="Toggle the visibility of mean lines.")

            # Create subplots with 2 rows per feature: one for original data, one for mean
            fig_time = make_subplots(
                rows=2*num_features,  # Two rows per feature
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[
                    f"{feature['display_name']}" for feature in features
                ] + [
                    f"{feature['display_name']} - Mean" for feature in features
                ]
            )

            # Calculate rolling means for each feature
            for feature in features:
                df[f"{feature['column']}_mean"] = df[feature['column']].rolling(window=window_size, min_periods=1).mean()

            for i, feature in enumerate(features, start=1):
                # Original Feature Plot on odd rows
                fig_time.add_trace(
                    go.Scatter(
                        x=df["Time_unit"],
                        y=df[feature["column"]],
                        mode="lines",
                        name=feature["display_name"],
                        line=dict(color=feature["color"]),
                    ),
                    row=2*i-1,
                    col=1,
                )
                # Update y-axis for original feature
                fig_time.update_yaxes(title_text=feature["display_name"], row=2*i-1, col=1)

                # Mean Feature Plot on even rows
                if show_means:
                    fig_time.add_trace(
                        go.Scatter(
                            x=df["Time_unit"],
                            y=df[f"{feature['column']}_mean"],
                            mode="lines",
                            name=f"{feature['display_name']} Mean",
                            line=dict(color=feature["color"], dash="dash"),
                        ),
                        row=2*i,
                        col=1,
                    )
                    # Update y-axis for mean feature
                    fig_time.update_yaxes(title_text=f"{feature['display_name']} Mean", row=2*i, col=1)

            fig_time.update_layout(
                xaxis_title=f"Time ({time_unit})",
                height=300 * 2 * num_features,  # 300 pixels per subplot
                showlegend=False,
                title_text="Features over Time (Original and Mean Values)",
            )

            st.plotly_chart(fig_time, use_container_width=True)

            # Provide explanations and annotations
            st.write(
                """
                **Interpretation Guide:**

                - **Advance Rate**: Indicates the speed at which the machine is advancing. Fluctuations may indicate changes in ground conditions or operational parameters.
                - **Penetration Rate**: Calculated as Advance Rate divided by Revolution. Reflects how efficiently the machine penetrates the material per revolution.
                - **Thrust Force**: Represents the force applied at the cutting head. High values may indicate hard ground or potential mechanical issues.
                - **Thrust Force per Cutting Ring**: This metric normalizes the thrust force by the number of cutting rings, providing insight into the load per ring.
                - **Revolution**: The rotational speed of the cutting head. Variations can affect penetration rate and torque.
                - **Working Pressure**: The pressure at which the machine is operating. Sudden changes might indicate anomalies or operational adjustments.

                Use the visualizations to monitor trends and identify any unusual patterns that may require further investigation.
                """
            )

            # Download buttons for analysis results
            st.sidebar.markdown("## Download Results")
            stats_df = pd.DataFrame(
                {
                    "RPM": df[revolution_col].describe(),
                    "Calculated Torque": df["Calculated torque [kNm]"].describe(),
                    "Working Pressure": df[pressure_col].describe(),
                    "Advance Rate": df[advance_rate_col].describe(),
                    "Penetration Rate (Calculated)": df["Calculated Penetration Rate"].describe(),
                    "Thrust Force": df[thrust_force_col].describe(),
                    "Thrust Force per Cutting Ring": df["Thrust Force per Cutting Ring"].describe(),
                }
            )
            st.sidebar.markdown(
                get_table_download_link(
                    stats_df, "advanced_statistical_analysis.csv", "Download Statistical Analysis"
                ),
                unsafe_allow_html=True,
            )

# Correct the main function call
if __name__ == "__main__":
    advanced_page()


if __name__ == "__main__":
    main()
