import streamlit as st
import numpy as np
import pandas as pd
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import csv
from io import StringIO
import chardet

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_get_loc(columns, col_name):
    """
    Safely get the index of a column name in a DataFrame's columns.

    Args:
        columns: The columns of the DataFrame (df.columns).
        col_name: The column name to find.

    Returns:
        The index of the column if found; otherwise, 0.
    """
    try:
        return columns.get_loc(col_name)
    except KeyError:
        return 0  # Default to the first column if not found

def load_data(file, file_type):
    """
    Enhanced function to load CSV and Excel files with robust error handling
    and multiple format support.
    
    Args:
        file: File object from st.file_uploader
        file_type: String indicating file type ('csv' or 'xlsx')
        
    Returns:
        pandas.DataFrame or None if loading fails
    """
    encoding = None
    delimiter = None  # Initialize delimiter
    try:
        if file_type == 'csv':
            # Read file content and detect encoding
            file_content = file.read()
            detected = chardet.detect(file_content)
            encoding = detected['encoding']
            
            # Convert bytes to string using detected encoding
            try:
                content_str = file_content.decode(encoding)
            except UnicodeDecodeError:
                # Fallback to common encodings if detection fails
                for enc in ['utf-8', 'iso-8859-1', 'latin1', 'cp1252']:
                    try:
                        content_str = file_content.decode(enc)
                        encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Unable to decode file with any common encoding")

            # Create StringIO object for pandas to read
            string_data = StringIO(content_str)
            
            # Detect delimiter using csv.Sniffer
            try:
                dialect = csv.Sniffer().sniff(content_str[:4096])
                delimiter = dialect.delimiter
            except:
                # Try common delimiters if sniffer fails
                delimiters = [',', ';', '\t', '|']
                max_columns = 0
                best_delimiter = None
                
                for delim in delimiters:
                    try:
                        string_data.seek(0)
                        test_df = pd.read_csv(string_data, sep=delim, nrows=5)
                        num_columns = len(test_df.columns)
                        
                        if num_columns > max_columns:
                            max_columns = num_columns
                            best_delimiter = delim
                    except:
                        continue
                
                if best_delimiter is None:
                    raise ValueError("Unable to determine CSV delimiter")
                
                delimiter = best_delimiter

            # Try reading with detected parameters
            try:
                string_data.seek(0)
                df = pd.read_csv(
                    string_data,
                    sep=delimiter,
                    encoding=encoding,
                    on_bad_lines='warn',
                    low_memory=False,
                    decimal=',',  # Handle European number format
                    thousands='.'  # Handle European number format
                )
                
                # Validate the DataFrame
                if df.empty:
                    raise ValueError("The resulting DataFrame is empty")
                
                if len(df.columns) == 1:
                    # If only one column, the delimiter might be wrong
                    raise ValueError("Only one column detected, delimiter might be incorrect")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Remove any completely empty rows or columns
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                return df

            except Exception as e:
                st.error(f"Error reading CSV with detected parameters: {str(e)}")
                raise

        elif file_type == 'xlsx':
            try:
                df = pd.read_excel(
                    file,
                    engine='openpyxl',
                    na_values=['NA', 'N/A', ''],  # Common NA values
                    keep_default_na=True
                )
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Remove any completely empty rows or columns
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                return df
                
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                raise

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        # Log additional debug information only if applicable
        st.error("Debug information:")
        if file_type == 'csv':
            st.error(f"Detected encoding: {encoding}")
            st.error(f"Detected delimiter: {delimiter}")
        else:
            st.warning("Additional debug information is not available for non-CSV files.")
        return None

# Update the sensor column map with more potential column names
sensor_column_map = {
    "pressure": ["Working pressure [bar]", "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26", "Pression [bar]", "Presión [bar]", "Pressure", "Pressure [bar]", "Working Pressure","cutting wheel.MPU1WPr","MPU1WPr"],
    "revolution": ["Revolution [rpm]", "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30", "Vitesse [rpm]", "Revoluciones [rpm]", "RPM", "Speed", "Rotation Speed","cutting wheel.CWSpeed","CWSpeed","cutting wheel"],
    "time": ["Time", "Timestamp", "DateTime", "Date", "Zeit", "Relativzeit", "Uhrzeit", "Datum", "ts(utc)"],
    "advance_rate": ["Advance Rate", "Vorschubgeschwindigkeit", "Avance", "Rate of Penetration", "ROP", "Advance [m/min]", "Advance [mm/min]","VTgeschw_Z","VTgeschw"],
    "thrust_force": ["Thrust Force", "Thrust", "Vorschubkraft", "Force", "Force at Cutting Head", "Thrust Force [kN]","15_thrust cylinder.TZylGrABCDForce","thrust cylinder.TZylGrABCDForce","TZylGrABCDForce"],
    "distance": ["Distance", "Chainage", "Position", "Kette", "Station","V34_TL_SR_m_Z","TL_SR_m_Z","SR_m_Z","Weg","weg"]
}

def find_sensor_columns(df):
    found_columns = {}
    for sensor, possible_names in sensor_column_map.items():
        for name in possible_names:
            # Case-insensitive and whitespace-stripped matching
            for col in df.columns:
                if name.strip().lower() == col.strip().lower():
                    found_columns[sensor] = col
                    break
        # If still not found, attempt partial matches
        if sensor not in found_columns:
            for col in df.columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    found_columns[sensor] = col
                    break
    return found_columns

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
    # Filter the DataFrame for the selected machine type
    machine_rows = specs_df[specs_df['Projekt'] == machine_type]
    if machine_rows.empty:
        st.error(f"Machine type '{machine_type}' not found in the specifications file.")
        return None

    # Extract the first matching row
    machine_data = machine_rows.iloc[0]

    # Define possible column names for each parameter
    n1_names = ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Max RPM']
    n2_names = ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Min RPM']
    m_cont_names = ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque']
    m_max_names = ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 'Max Torque']
    torque_constant_names = ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]', 'Torque Constant']

    # Function to find the correct column name
    def find_column(possible_names):
        for name in possible_names:
            if name in machine_data.index:
                return name
        return None

    # Attempt to find each parameter
    n1_col = find_column(n1_names)
    n2_col = find_column(n2_names)
    m_cont_col = find_column(m_cont_names)
    m_max_col = find_column(m_max_names)
    torque_constant_col = find_column(torque_constant_names)

    # Collect missing parameters
    missing_params = []
    if n1_col is None:
        missing_params.append('n1 (Maximum RPM)')
    if n2_col is None:
        missing_params.append('n2 (Minimum RPM)')
    if m_cont_col is None:
        missing_params.append('M_cont_value (Continuous Torque)')
    if m_max_col is None:
        missing_params.append('M_max_Vg1 (Maximum Torque)')
    if torque_constant_col is None:
        missing_params.append('torque_constant')

    # If any parameters are missing, return None
    if missing_params:
        st.error(f"Missing parameters for machine '{machine_type}': {', '.join(missing_params)}. Please check the specifications file.")
        return None

    # Return the found parameters
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

# Helper Function Defined Outside
def format_timedelta(td):
    """
    Convert a pandas Timedelta to a human-readable string.
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

            # After selecting the machine type
            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved. Please ensure the specifications file contains all required parameters for the selected machine.")
                st.stop()


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
            pressure_col = st.selectbox(
                "Select Pressure Column",
                options=df.columns,
                index=safe_get_loc(df.columns, sensor_columns.get('pressure', df.columns[0]))
            )
            revolution_col = st.selectbox(
                "Select Revolution Column",
                options=df.columns,
                index=safe_get_loc(df.columns, sensor_columns.get('revolution', df.columns[0]))
            )


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


def advanced_page():
    """Full implementation of the advanced analysis page - Part 1: Initial Setup and File Loading"""
    
    st.title("Advanced Analysis")

    # File uploaders for batch data
    raw_data_file = st.file_uploader(
        "Upload Raw Data (CSV or XLSX)", 
        type=["csv", "xlsx"]
    )
    machine_specs_file = st.file_uploader(
        "Upload Machine Specifications: XLSX (MM-Baureihenliste) or CSV format accepted",
        type=["xlsx", "csv"],
    )

    # Handle missing files
    if not all([raw_data_file, machine_specs_file]):
        st.warning("Please upload both Raw Data and Machine Specifications files.")
        return

    try:
        # Load machine specifications
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

        # Machine selection
        selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

        # Get machine parameters
        machine_params = get_machine_params(machine_specs, selected_machine)
        if not machine_params:
            st.error("Machine parameters could not be retrieved.")
            st.stop()

        # Display machine parameters table
        params_df = pd.DataFrame([machine_params])
        styled_table = params_df.style.set_table_styles([
            {
                "selector": "th",
                "props": [("border", "2px solid black"), ("padding", "5px")],
            },
            {
                "selector": "td",
                "props": [("border", "2px solid black"), ("padding", "5px")],
            },
            {"selector": "", "props": [("border-collapse", "collapse")]},
        ]).to_html()

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

# Part 2: Parameter Settings and Data Loading
        # Set up sidebar parameters for analysis
        st.sidebar.header("Parameter Settings")
        P_max = st.sidebar.number_input(
            "Maximum power (kW)", 
            value=132.0, 
            min_value=1.0, 
            max_value=500.0,
            help="Maximum power output of the machine"
        )
        nu = st.sidebar.number_input(
            "Efficiency coefficient", 
            value=0.7, 
            min_value=0.1, 
            max_value=1.0,
            help="Machine efficiency factor between 0.1 and 1.0"
        )
        anomaly_threshold = st.sidebar.number_input(
            "Anomaly threshold (bar)", 
            value=250, 
            min_value=100, 
            max_value=500,
            help="Pressure threshold for identifying anomalies"
        )
        num_cutting_rings = st.sidebar.number_input(
            "Number of Cutting Rings", 
            value=1, 
            min_value=1, 
            max_value=100,
            help="Number of cutting rings on the machine"
        )

        # Load and process raw data
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            # Find sensor columns with automatic detection
            sensor_columns = find_sensor_columns(df)

            # Column selection interface
            st.subheader("Select Sensor Columns")
            
            # Time column type selection
            time_column_type_user = st.selectbox(
                "Select Time Column Type",
                options=["Numeric", "Datetime"],
                index=0,
                help="Choose how to interpret the Time column."
            )

            # Time Column
            if "time" in sensor_columns and sensor_columns["time"] in df.columns:
                default_time_col = sensor_columns["time"]
            else:
                st.warning("Time column not found automatically. Please select it manually.")
                default_time_col = df.columns[0]
            time_col = st.selectbox(
                "Select Time Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_time_col)
            )

            # Pressure Column
            if "pressure" in sensor_columns and sensor_columns["pressure"] in df.columns:
                default_pressure_col = sensor_columns["pressure"]
            else:
                st.warning("Pressure column not found automatically. Please select it manually.")
                default_pressure_col = df.columns[0]
            pressure_col = st.selectbox(
                "Select Pressure Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_pressure_col)
            )

            # Revolution Column
            if "revolution" in sensor_columns and sensor_columns["revolution"] in df.columns:
                default_revolution_col = sensor_columns["revolution"]
            else:
                st.warning("Revolution column not found automatically. Please select it manually.")
                default_revolution_col = df.columns[0]
            revolution_col = st.selectbox(
                "Select Revolution Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_revolution_col)
            )

            # Advance Rate Column
            if "advance_rate" in sensor_columns and sensor_columns["advance_rate"] in df.columns:
                default_advance_rate_col = sensor_columns["advance_rate"]
            else:
                st.warning("Advance rate column not found automatically. Please select it manually.")
                default_advance_rate_col = df.columns[0]
            advance_rate_col = st.selectbox(
                "Select Advance Rate Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_advance_rate_col)
            )

            # Thrust Force Column
            if "thrust_force" in sensor_columns and sensor_columns["thrust_force"] in df.columns:
                default_thrust_force_col = sensor_columns["thrust_force"]
            else:
                st.warning("Thrust force column not found automatically. Please select it manually.")
                default_thrust_force_col = df.columns[0]
            thrust_force_col = st.selectbox(
                "Select Thrust Force Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_thrust_force_col)
            )

            # Distance/Chainage Column
            if "distance" in sensor_columns and sensor_columns["distance"] in df.columns:
                default_distance_col = sensor_columns["distance"]
            else:
                st.warning("Distance/Chainage column not found automatically. Please select it manually.")
                default_distance_col = df.columns[0]
            distance_col = st.selectbox(
                "Select Distance/Chainage Column",
                options=df.columns,
                index=safe_get_loc(df.columns, default_distance_col)
            )

# Part 3: Data Processing and Time Series Handling
            
            # Ensure distance column is properly processed
            df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
            if df[distance_col].isnull().all():
                st.error(
                    f"The selected distance/chainage column '{distance_col}' cannot be converted to numeric values."
                )
                return

            # Handle missing distance values
            missing_distance = df[distance_col].isnull().sum()
            if missing_distance > 0:
                st.warning(f"There are {missing_distance} missing values in the distance/chainage column. These rows will be dropped.")
                df = df.dropna(subset=[distance_col])

            # Display the maximum distance value for validation
            max_distance_value = df[distance_col].max()
            st.write(f"**Maximum value in the distance/chainage column (`{distance_col}`):** {max_distance_value}")

            # Time column processing based on user selection
            if time_column_type_user == "Numeric":
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                if df[time_col].isnull().all():
                    st.error(
                        f"The selected time column '{time_col}' cannot be converted to numeric values."
                    )
                    return
                time_column_type = 'numeric'

                # Assign Time_unit for plotting
                df["Time_unit"] = df[time_col]

                # Sort the dataframe by Time_unit
                df = df.sort_values("Time_unit")

                # Calculate min and max time
                min_time_unit = df["Time_unit"].min()
                max_time_unit = df["Time_unit"].max()

                # Display the time range in numeric format
                st.write(f"**Data time range:** {min_time_unit:.2f} to {max_time_unit:.2f} units")

                # Create the time range slider
                time_range = st.slider(
                    "Select Time Range",
                    min_value=float(min_time_unit),
                    max_value=float(max_time_unit),
                    value=(float(min_time_unit), float(max_time_unit)),
                    format="%.2f",
                )

                # Filter data based on the selected time range
                df = df[(df["Time_unit"] >= time_range[0]) & (df["Time_unit"] <= time_range[1])]

            else:
                # Handle datetime format
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    if df[time_col].isnull().all():
                        raise ValueError("All values in the time column are NaT after parsing.")
                    time_column_type = 'datetime'
                except Exception as e:
                    # Try to convert to numeric if datetime fails
                    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
                    if df[time_col].isnull().all():
                        st.error(
                            f"The selected time column '{time_col}' cannot be converted to datetime or numeric values."
                        )
                        return
                    time_column_type = 'numeric'

                if time_column_type == 'datetime':
                    # Assign Time_unit for plotting
                    df["Time_unit"] = df[time_col]

                    # Sort the dataframe by Time_unit
                    df = df.sort_values("Time_unit")

                    # Calculate min and max time
                    min_time_unit = df["Time_unit"].min().to_pydatetime()
                    max_time_unit = df["Time_unit"].max().to_pydatetime()

                    # Display the time range
                    st.write(f"**Data time range:** {min_time_unit} to {max_time_unit}")

                    # Define a reasonable step for the slider (e.g., one second)
                    time_step = timedelta(seconds=1)

                    # Create the time range slider
                    time_range = st.slider(
                        "Select Time Range",
                        min_value=min_time_unit,
                        max_value=max_time_unit,
                        value=(min_time_unit, max_time_unit),
                        format="YYYY-MM-DD HH:mm:ss",
                        step=time_step,
                    )

                    # Filter data based on the selected time range
                    df = df[(df["Time_unit"] >= time_range[0]) & (df["Time_unit"] <= time_range[1])]

            # Convert all measurement columns to numeric
            for col in [
                pressure_col,
                revolution_col,
                advance_rate_col,
                thrust_force_col,
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Remove rows with NaN values in essential columns
            df = df.dropna(
                subset=[pressure_col, revolution_col, advance_rate_col, thrust_force_col]
            )

            # Remove rows where revolution is zero to avoid division by zero
            df = df[df[revolution_col] != 0]

            # Calculate derived metrics
            df["Calculated Penetration Rate"] = (
                df[advance_rate_col] / df[revolution_col]
            )

            df["Thrust Force per Cutting Ring"] = df[thrust_force_col] / num_cutting_rings

            # RPM Statistics for plot scaling
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

# Part 4: Torque Calculations and Analysis
            
            # Filter data points between n2 and n1 rpm based on machine parameters
            n2 = machine_params.get("n2", df[revolution_col].min())
            n1 = machine_params.get("n1", df[revolution_col].max())
            df = df[
                (df[revolution_col] >= n2)
                & (df[revolution_col] <= n1)
            ]

            # Define the torque calculation function for each data point
            def calculate_torque_wrapper(row):
                working_pressure = row[pressure_col]
                current_speed = row[revolution_col]

                if current_speed < machine_params["n1"]:
                    # Standard torque calculation when speed is below n1
                    torque = working_pressure * machine_params["torque_constant"]
                else:
                    # Adjusted torque calculation when speed exceeds n1
                    torque = (
                        (machine_params["n1"] / current_speed)
                        * machine_params["torque_constant"]
                        * working_pressure
                    )

                return round(torque, 2)

            # Calculate torque for each row in the dataset
            df["Calculated torque [kNm]"] = df.apply(
                calculate_torque_wrapper, axis=1
            )

            # Calculate statistical measures for outlier detection
            # Using 10th and 90th percentiles for more robust outlier detection
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

            # Identify anomalies based on pressure threshold
            df["Is_Anomaly"] = df[pressure_col] >= anomaly_threshold

            # Calculate maximum theoretical torque based on power and efficiency
            def M_max_Vg2(rpm):
                """Calculate maximum theoretical torque for given RPM values"""
                return np.minimum(
                    machine_params["M_max_Vg1"],  # Maximum torque limit
                    (P_max * 60 * nu) / (2 * np.pi * rpm)  # Power-based torque limit
                )

            # Calculate critical points for torque curves
            elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params["M_max_Vg1"])
            elbow_rpm_cont = (
                P_max * 60 * nu
            ) / (2 * np.pi * machine_params["M_cont_value"])

            # Generate RPM values for torque curve plotting
            rpm_curve = np.linspace(0.1, machine_params["n1"], 1000)

            # Create subplot for torque analysis
            fig = make_subplots(rows=1, cols=1)

            # Add continuous torque limit curve
            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= elbow_rpm_cont],
                    y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], 
                                 machine_params["M_cont_value"]),
                    mode="lines",
                    name="M cont Max [kNm]",
                    line=dict(color="green", width=2),
                )
            )

            # Add maximum torque limit curve
            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= elbow_rpm_max],
                    y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], 
                                 machine_params["M_max_Vg1"]),
                    mode="lines",
                    name="M max Vg1 [kNm]",
                    line=dict(color="red", width=2),
                )
            )

            # Add power-limited torque curve
            fig.add_trace(
                go.Scatter(
                    x=rpm_curve[rpm_curve <= machine_params["n1"]],
                    y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params["n1"]]),
                    mode="lines",
                    name="M max Vg2 [kNm]",
                    line=dict(color="red", width=2, dash="dash"),
                )
            )

            # Calculate y-values for vertical lines at critical points
            y_max_vg2 = M_max_Vg2(
                np.array([
                    elbow_rpm_max,
                    elbow_rpm_cont,
                    machine_params["n1"],
                ])
            )

            # Add vertical reference lines at critical points
            fig.add_trace(
                go.Scatter(
                    x=[elbow_rpm_max, elbow_rpm_max],
                    y=[0, y_max_vg2[0]],
                    mode="lines",
                    line=dict(color="purple", width=1, dash="dot"),
                    showlegend=False,
                    name="Elbow RPM Max"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[elbow_rpm_cont, elbow_rpm_cont],
                    y=[0, y_max_vg2[1]],
                    mode="lines",
                    line=dict(color="orange", width=1, dash="dot"),
                    showlegend=False,
                    name="Elbow RPM Cont"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[machine_params["n1"], machine_params["n1"]],
                    y=[0, y_max_vg2[2]],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    showlegend=False,
                    name="Maximum RPM (n1)"
                )
            )

# Part 5: Data Point Visualization and Statistical Analysis
            
            # Separate data into different categories for visualization
            normal_data = df[~df["Is_Anomaly"]]
            anomaly_data = df[df["Is_Anomaly"]]

            # Identify outlier data points
            torque_outlier_data = df[df["Calculated torque [kNm]"].isin(torque_outliers)]
            rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

            # Add normal operating points to the plot
            fig.add_trace(
                go.Scatter(
                    x=normal_data[revolution_col],
                    y=normal_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="Normal Operation",
                    marker=dict(
                        color=normal_data["Calculated torque [kNm]"],
                        colorscale="Viridis",
                        size=8,
                        showscale=True,
                        colorbar=dict(
                            title="Torque [kNm]",
                            x=1.1
                        )
                    )
                )
            )

            # Add anomaly points (high pressure) to the plot
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data[revolution_col],
                    y=anomaly_data["Calculated torque [kNm]"],
                    mode="markers",
                    name=f"Pressure Anomaly (≥{anomaly_threshold} bar)",
                    marker=dict(
                        color="red",
                        symbol="x",
                        size=10,
                        line=dict(width=2)
                    )
                )
            )

            # Add torque outliers to the plot
            fig.add_trace(
                go.Scatter(
                    x=torque_outlier_data[revolution_col],
                    y=torque_outlier_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="Torque Outliers",
                    marker=dict(
                        color="orange",
                        symbol="diamond",
                        size=10,
                        line=dict(width=2)
                    )
                )
            )

            # Add RPM outliers to the plot
            fig.add_trace(
                go.Scatter(
                    x=rpm_outlier_data[revolution_col],
                    y=rpm_outlier_data["Calculated torque [kNm]"],
                    mode="markers",
                    name="RPM Outliers",
                    marker=dict(
                        color="purple",
                        symbol="square",
                        size=10,
                        line=dict(width=2)
                    )
                )
            )

            # Add reference lines for statistical thresholds
            fig.add_hline(
                y=torque_upper_whisker,
                line_dash="dash",
                line_color="gray",
                annotation_text="Torque Upper Whisker (90th Percentile)",
                annotation_position="right"
            )
            fig.add_hline(
                y=torque_lower_whisker,
                line_dash="dot",
                line_color="gray",
                annotation_text="Torque Lower Whisker (10th Percentile)",
                annotation_position="right"
            )

            # Update plot layout with comprehensive formatting
            fig.update_layout(
                title=dict(
                    text=f"{selected_machine} - Advanced Torque Analysis",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Revolution [rpm]",
                yaxis_title="Torque [kNm]",
                xaxis=dict(
                    range=[0, x_axis_max],
                    gridcolor='lightgray',
                    showgrid=True
                ),
                yaxis=dict(
                    range=[0, max(60, df["Calculated torque [kNm]"].max() * 1.1)],
                    gridcolor='lightgray',
                    showgrid=True
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                ),
                width=1000,
                height=800,
                margin=dict(l=50, r=50, t=100, b=100),
                plot_bgcolor='white'
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Create comprehensive statistical summary
            st.header("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Statistics")
                stats_df = pd.DataFrame({
                    'Revolution [rpm]': df[revolution_col].describe(),
                    'Pressure [bar]': df[pressure_col].describe(),
                    'Torque [kNm]': df['Calculated torque [kNm]'].describe(),
                    'Penetration Rate': df['Calculated Penetration Rate'].describe()
                }).round(2)
                st.dataframe(stats_df)

            with col2:
                st.subheader("Operating Metrics")
                # Calculate percentage of anomalies
                anomaly_percentage = (df['Is_Anomaly'].sum() / len(df)) * 100
                st.write(f"Anomaly Percentage: {anomaly_percentage:.2f}%")
                
                # Calculate time in different operating ranges
                time_in_ranges = pd.cut(
                    df['Calculated torque [kNm]'],
                    bins=[0, machine_params['M_cont_value'], 
                         machine_params['M_max_Vg1'], float('inf')],
                    labels=['Normal', 'High', 'Critical']
                ).value_counts(normalize=True) * 100
                
                st.write("Time in Operating Ranges:")
                for range_name, percentage in time_in_ranges.items():
                    st.write(f"- {range_name}: {percentage:.2f}%")

# Part 6: Time Series Analysis, Distance Analysis, and Final Processing
            
            # Time Series Analysis Section
            st.header("Time Series Analysis")
            
            # Define the features to analyze with their display names and colors
            features = [
                {"column": advance_rate_col, "display_name": "Advance Rate", "color": "blue", "unit": "mm/min"},
                {"column": "Calculated Penetration Rate", "display_name": "Penetration Rate", "color": "green", "unit": "mm/rev"},
                {"column": thrust_force_col, "display_name": "Thrust Force", "color": "red", "unit": "kN"},
                {"column": "Thrust Force per Cutting Ring", "display_name": "Thrust Force per Ring", "color": "orange", "unit": "kN"},
                {"column": revolution_col, "display_name": "Revolution", "color": "purple", "unit": "rpm"},
                {"column": pressure_col, "display_name": "Working Pressure", "color": "cyan", "unit": "bar"},
            ]

            num_features = len(features)

            # Add rolling window size control
            window_size = st.sidebar.slider(
                "Rolling Window Size for Mean Calculation",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Larger window = smoother trend line"
            )

            # Toggle for mean lines visibility
            show_means = st.checkbox("Show Rolling Mean Values", value=True)

            # Calculate rolling means for each feature
            for feature in features:
                df[f"{feature['column']}_mean"] = df[feature['column']].rolling(
                    window=window_size,
                    min_periods=1
                ).mean()

            # Create time series subplots
            fig_time = make_subplots(
                rows=2*num_features,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=None
            )

            # Add traces for each feature
            for i, feature in enumerate(features, start=1):
                # Original data trace
                fig_time.add_trace(
                    go.Scatter(
                        x=df["Time_unit"],
                        y=df[feature["column"]],
                        mode="lines",
                        name=f"{feature['display_name']} ({feature['unit']})",
                        line=dict(color=feature["color"]),
                    ),
                    row=2*i-1,
                    col=1
                )
                
                # Update y-axis title for original data
                fig_time.update_yaxes(
                    title_text=f"{feature['display_name']} ({feature['unit']})",
                    row=2*i-1,
                    col=1
                )

                # Rolling mean trace
                if show_means:
                    fig_time.add_trace(
                        go.Scatter(
                            x=df["Time_unit"],
                            y=df[f"{feature['column']}_mean"],
                            mode="lines",
                            name=f"{feature['display_name']} Mean",
                            line=dict(
                                color=feature["color"],
                                dash="dash"
                            ),
                        ),
                        row=2*i,
                        col=1
                    )
                    
                    # Update y-axis title for rolling mean
                    fig_time.update_yaxes(
                        title_text=f"{feature['display_name']} - Rolling Mean",
                        row=2*i,
                        col=1
                    )

            # Update time series plot layout
            fig_time.update_layout(
                height=300 * 2 * num_features,
                showlegend=True,
                title_text="Features Over Time Analysis",
                xaxis_title="Time"
            )

            # Display time series plot
            st.plotly_chart(fig_time, use_container_width=True)

            # Distance-based Analysis
            st.header("Distance/Chainage Analysis")

            # Add Calculated Torque to features for distance analysis
            features_distance = features + [
                {"column": "Calculated torque [kNm]", "display_name": "Calculated Torque", "color": "magenta", "unit": "kNm"}
            ]
            num_features_distance = len(features_distance)

            # Sort data by distance for proper plotting
            df = df.sort_values(by=distance_col)

            # Calculate rolling means for distance-based analysis
            for feature in features_distance:
                df[f"{feature['column']}_distance_mean"] = df[feature['column']].rolling(
                    window=window_size,
                    min_periods=1
                ).mean()

            # Create distance analysis subplots
            fig_distance = make_subplots(
                rows=2*num_features_distance,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=None
            )

            # Add traces for distance-based analysis
            for i, feature in enumerate(features_distance, start=1):
                # Original data trace
                fig_distance.add_trace(
                    go.Scatter(
                        x=df[distance_col],
                        y=df[feature["column"]],
                        mode="lines",
                        name=f"{feature['display_name']} ({feature['unit']})",
                        line=dict(color=feature["color"]),
                    ),
                    row=2*i-1,
                    col=1
                )
                
                # Update y-axis title for original data
                fig_distance.update_yaxes(
                    title_text=f"{feature['display_name']} ({feature['unit']})",
                    row=2*i-1,
                    col=1
                )

                # Rolling mean trace
                if show_means:
                    fig_distance.add_trace(
                        go.Scatter(
                            x=df[distance_col],
                            y=df[f"{feature['column']}_distance_mean"],
                            mode="lines",
                            name=f"{feature['display_name']} Mean",
                            line=dict(
                                color=feature["color"],
                                dash="dash"
                            ),
                        ),
                        row=2*i,
                        col=1
                    )
                    
                    # Update y-axis title for rolling mean
                    fig_distance.update_yaxes(
                        title_text=f"{feature['display_name']} - Rolling Mean",
                        row=2*i,
                        col=1
                    )

            # Update distance plot layout
            fig_distance.update_layout(
                height=300 * 2 * num_features_distance,
                showlegend=True,
                title_text="Features Over Distance Analysis",
                xaxis_title="Distance/Chainage"
            )

            # Display distance analysis plot
            st.plotly_chart(fig_distance, use_container_width=True)

            # Add download options
            st.sidebar.header("Download Results")
            
            # Prepare data for download
            download_df = df[[
                distance_col, time_col, pressure_col, revolution_col,
                advance_rate_col, thrust_force_col,
                'Calculated torque [kNm]', 'Calculated Penetration Rate',
                'Thrust Force per Cutting Ring', 'Is_Anomaly'
            ]]
            
            # Create download link
            csv = download_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download Complete Analysis Results (CSV)</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        logging.error(f"Error in advanced_page: {str(e)}", exc_info=True)
        st.stop()






if __name__ == "__main__":
    main()
