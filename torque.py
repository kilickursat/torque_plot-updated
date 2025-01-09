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
import traceback

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



def convert_to_arrow_compatible(df):
    """
    Convert DataFrame to be fully Arrow-compatible by explicitly handling each data type.
    """
    converted_df = df.copy()
    
    for col in converted_df.columns:
        # Get the current dtype
        current_dtype = converted_df[col].dtype
        
        try:
            if pd.api.types.is_datetime64_any_dtype(current_dtype):
                # Convert datetime to int64 timestamp (nanoseconds since epoch)
                converted_df[col] = converted_df[col].astype(np.int64)
                
            elif current_dtype == 'object':
                # Try numeric conversion first
                try:
                    converted_df[col] = pd.to_numeric(converted_df[col], errors='raise')
                except (ValueError, TypeError):
                    # If numeric conversion fails, try datetime
                    try:
                        converted_df[col] = pd.to_datetime(converted_df[col], errors='raise')
                        # Convert datetime to int64 timestamp
                        converted_df[col] = converted_df[col].astype(np.int64)
                    except (ValueError, TypeError):
                        # If all else fails, convert to string
                        converted_df[col] = converted_df[col].astype(str)
                        
            elif current_dtype == 'bool':
                converted_df[col] = converted_df[col].astype(int)
                
            elif current_dtype == 'category':
                converted_df[col] = converted_df[col].astype(str)
                
            # Handle any remaining numeric types
            elif pd.api.types.is_numeric_dtype(current_dtype):
                # Ensure all numeric types are either int64 or float64
                if pd.api.types.is_integer_dtype(current_dtype):
                    converted_df[col] = converted_df[col].astype(np.int64)
                else:
                    converted_df[col] = converted_df[col].astype(np.float64)
                    
        except Exception as e:
            st.warning(f"Could not convert column '{col}' automatically. Converting to string. Error: {str(e)}")
            converted_df[col] = converted_df[col].astype(str)
    
    return converted_df

import warnings
warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds in conversion")

def load_data(file, file_type):
    try:
        raw_content = file.read()
        encoding = chardet.detect(raw_content)['encoding'] or 'utf-8'
        content_str = raw_content.decode(encoding)
        
        # Detect if this is data 21 format by checking first line
        is_data21_format = 'ts(utc)' in content_str.split('\n')[0]
        
        if file_type == 'csv':
            delimiter = ';' if ';' in content_str.split('\n')[0] else ','
            
            # Special handling for data 21 format
            if is_data21_format:
                cleaned_lines = []
                header = True
                for line in content_str.split('\n'):
                    if line.strip():
                        if header:
                            cleaned_lines.append(line)
                            header = False
                        else:
                            parts = line.split(delimiter)
                            if len(parts) > 1:
                                # Keep timestamp as is, convert other values
                                cleaned_parts = [parts[0]] + [p.replace(',', '.') for p in parts[1:]]
                                cleaned_lines.append(delimiter.join(cleaned_parts))
                
                df = pd.read_csv(
                    StringIO('\n'.join(cleaned_lines)),
                    sep=delimiter,
                    dtype={'ts(utc)': str},
                    decimal='.',
                    thousands=None,
                    na_values=['NA', 'N/A', '', 'nan', 'NaN', 'null', 'NULL'],
                    skipinitialspace=True
                )
                
                # Convert UTC timestamp
                if 'ts(utc)' in df.columns:
                    df['ts(utc)'] = pd.to_datetime(df['ts(utc)']).dt.floor('s')
                    df['ts(utc)'] = (df['ts(utc)'].astype('int64') // 10**9).astype('int64')
            
            else:
                # Original format handling
                df = pd.read_csv(
                    StringIO(content_str),
                    sep=delimiter,
                    decimal=',',
                    thousands=None,
                    na_values=['NA', 'N/A', '', 'nan', 'NaN', 'null', 'NULL'],
                    skipinitialspace=True
                )
                
                # Handle date and time if present
                if all(col in df.columns for col in ['Datum', 'Uhrzeit']):
                    df['timestamp'] = pd.to_datetime(
                        df['Datum'] + ' ' + df['Uhrzeit'],
                        format='%d.%m.%Y %H:%M:%S',
                        errors='coerce'
                    )
            
            # Convert numeric columns
            numeric_cols = df.columns.difference(['ts(utc)', 'Datum', 'Uhrzeit', 'timestamp'])
            for col in numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    continue
            
            df.columns = df.columns.str.strip()
            df = df.dropna(how='all')

        elif file_type == 'xlsx':
            file.seek(0)
            df = pd.read_excel(
                file,
                engine='openpyxl',
                na_values=['NA', 'N/A', '', 'nan', 'NaN', 'null', 'NULL']
            )

            if 'ts(utc)' in df.columns:
                df['ts(utc)'] = pd.to_datetime(df['ts(utc)']).dt.floor('s')
                df['ts(utc)'] = (df['ts(utc)'].astype('int64') // 10**9).astype('int64')

            numeric_cols = df.columns.difference(['ts(utc)'])
            for col in numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    continue

            df.columns = df.columns.str.strip()
            df = df.dropna(how='all')

        if df.empty:
            raise ValueError("DataFrame is empty after processing")
            
        file.seek(0)
        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
        
sensor_column_map = {
    "pressure": [
        "Working pressure [bar]", 
        "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26",
        "V13_SR_ArbDr_Z",
        "Pressure",
        "Pressure [bar]", 
        "Working Pressure",
        "cutting wheel.MPU1WPr",
        "MPU1WPr",
        "AzV.V13_SR_ArbDr_Z",
        "Pression [bar]", 
        "Presión [bar]"
    ],
    "revolution": [
        "Revolution [rpm]",
        "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30",
        "V13_SR_Drehz_nach_Abgl_Z",
        "Vitesse [rpm]",
        "Revoluciones [rpm]",
        "RPM",
        "Speed",
        "Rotation Speed",
        "cutting wheel.CWSpeed",
        "CWSpeed",
        "cutting wheel",
        "AzV.V13_SR_Drehz_nach_Abgl_Z"
    ],
    "time": [
        "Time",
        "Timestamp",
        "DateTime",
        "Date",
        "Zeit",
        "Relativzeit",
        "Uhrzeit",
        "Datum",
        "ts(utc)"
    ],
    "advance_rate": [
        "Advance Rate",
        "V34_VTgeschw_Z",
        "VTgeschw_Z",
        "VTgeschw",
        "Vorschubgeschwindigkeit",
        "Avance",
        "Rate of Penetration",
        "ROP",
        "Advance [m/min]",
        "Advance [mm/min]"
    ],
    "thrust_force": [
        "Thrust Force",
        "V15_VTP_Kraft_max_V",
        "Thrust",
        "Vorschubkraft",
        "Force",
        "Force at Cutting Head",
        "Thrust Force [kN]",
        "15_thrust cylinder.TZylGrABCDForce",
        "thrust cylinder.TZylGrABCDForce",
        "TZylGrABCDForce"
    ],
    "distance": [
        "Distance",
        "V15_Dehn_Weg_ges_Z",
        "Chainage",
        "Position",
        "Kette",
        "Station",
        "V34_TL_SR_m_Z",
        "TL_SR_m_Z",
        "SR_m_Z",
        "Weg",
        "weg"
    ]
}

def find_sensor_columns(df):
    found_columns = {}
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    for sensor, possible_names in sensor_column_map.items():
        # First try exact matches (case-insensitive)
        for name in possible_names:
            name_lower = name.strip().lower()
            for col in df.columns:
                if name_lower == col.lower():
                    found_columns[sensor] = col
                    break
            if sensor in found_columns:
                break
                
        # If no exact match, try partial matches
        if sensor not in found_columns:
            for col in df.columns:
                col_lower = col.lower()
                if any(name.strip().lower() in col_lower for name in possible_names):
                    found_columns[sensor] = col
                    break
                    
    return found_columns
    
def handle_time_column(df, time_col):
    try:
        if time_col == 'ts(utc)':
            return pd.to_datetime(df[time_col]).astype(np.int64) // 10**9
        elif time_col in ['Relativzeit', 'Datum', 'Uhrzeit']:
            return pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype('int64')
        else:
            return pd.Series(range(len(df)), dtype='int64')
    except Exception as e:
        st.error(f"Time column processing error: {str(e)}")
        return pd.Series(range(len(df)), dtype='int64')

def update_plot_parameters(df, revolution_col):
    rpm_stats = df[revolution_col].describe()
    x_axis_max = min(rpm_stats['max'] * 1.2, 15)  # Limit x-axis for Dataset 2
    return x_axis_max
        
def load_machine_specs(file, file_type):
    """Load and validate machine specifications with improved error handling."""
    try:
        if file_type == 'xlsx':
            # Try different sheets if available
            try:
                xl = pd.ExcelFile(file)
                if len(xl.sheet_names) > 1:
                    sheet_name = st.selectbox("Select sheet:", xl.sheet_names)
                else:
                    sheet_name = xl.sheet_names[0]
                
                specs_df = pd.read_excel(file, sheet_name=sheet_name)
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
                
        elif file_type == 'csv':
            try:
                # Try different encodings
                encodings = ['utf-8', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        specs_df = pd.read_csv(file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not read CSV file with any supported encoding")
                    return None
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return None
        else:
            st.error("Unsupported file type")
            return None
        
        # Clean column names
        specs_df.columns = specs_df.columns.str.strip()
        
        # Check for required column
        if 'Projekt' not in specs_df.columns:
            # Try to find alternative column names
            possible_names = ['Project', 'Machine', 'Machine Type', 'Type', 'Model']
            found = False
            for name in possible_names:
                if name in specs_df.columns:
                    specs_df.rename(columns={name: 'Projekt'}, inplace=True)
                    found = True
                    st.info(f"Using '{name}' as the machine type column")
                    break
            
            if not found:
                st.error("Required machine type column not found. Available columns:")
                st.write(specs_df.columns.tolist())
                return None
        
        # Remove any completely empty rows or columns
        specs_df = specs_df.dropna(how='all').dropna(axis=1, how='all')
        
        # Display validation info
        st.write("Loaded specifications data preview:")
        st.write(specs_df.head())
        st.write("Available columns:", specs_df.columns.tolist())
        
        return specs_df
        
    except Exception as e:
        st.error(f"Error loading machine specifications: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def get_machine_params(specs_df, machine_type):
    """Extract and validate machine parameters."""
    try:
        # Add debug information
        st.write("Debug: Available columns in specs_df:", specs_df.columns.tolist())
        st.write("Debug: First few rows of specs_df:")
        st.write(specs_df.head())
        
        # Normalize machine type strings for comparison
        specs_df['Normalized_Projekt'] = specs_df['Projekt'].str.replace('-', '').str.lower().str.strip()
        normalized_type = machine_type.replace('-', '').lower().strip()
        
        st.write("Debug: Looking for normalized machine type:", normalized_type)
        st.write("Debug: Available normalized machine types:", specs_df['Normalized_Projekt'].unique().tolist())
        
        # Filter for machine type using normalized comparison
        machine_rows = specs_df[specs_df['Normalized_Projekt'] == normalized_type]
        
        if machine_rows.empty:
            st.error(f"Machine type '{machine_type}' not found in specifications after normalization")
            # Try partial matching
            for idx, row in specs_df.iterrows():
                if normalized_type in row['Normalized_Projekt'] or row['Normalized_Projekt'] in normalized_type:
                    st.warning(f"Found partial match: {row['Projekt']}")
                    machine_rows = specs_df.iloc[[idx]]
                    break
            
            if machine_rows.empty:
                return None
        
        # Get the first matching row
        machine_data = machine_rows.iloc[0]
        st.write("Debug: Found machine data:", machine_data.to_dict())
        
        # Define parameter mappings with additional variations
        param_mappings = {
            'n1': ['n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Max RPM', 'n1', 'N1', 'n1', 'N1[rpm]', 'N1 [rpm]'],
            'n2': ['n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Min RPM', 'n2', 'N2', 'n2', 'N2[rpm]', 'N2 [rpm]'],
            'M_cont_value': ['M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque',
                            'M dauer', 'Mdauer', 'M_cont', 'M(cont)', 'M_cont[kNm]', 'M_cont [kNm]'],
            'M_max_Vg1': ['M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 'Max Torque',
                         'Mmax', 'M_max', 'M max[kNm]', 'M_max [kNm]'],
            'torque_constant': ['Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]',
                              'Torque Constant', 'Torque_Constant', 'TorqueConstant', 'TC[kNm/bar]', 'TC [kNm/bar]']
        }
        
        params = {}
        found_params = []
        missing_params = []
        
        # Try to find each parameter
        for param, possible_names in param_mappings.items():
            found = False
            for name in possible_names:
                if name in machine_data.index:
                    try:
                        value = machine_data[name]
                        # Handle different number formats
                        if isinstance(value, str):
                            # Remove any units in brackets and convert commas to dots
                            value = value.split('[')[0].replace(',', '.').strip()
                        
                        params[param] = float(value)
                        found = True
                        found_params.append(f"{param}: {params[param]} (from {name})")
                        break
                    except (ValueError, TypeError) as e:
                        st.warning(f"Could not convert {name} value '{value}' to float: {str(e)}")
                        continue
            
            if not found:
                missing_params.append(f"{param} (tried: {', '.join(possible_names)})")
        
        # Log found and missing parameters
        if found_params:
            st.info("Found parameters:\n" + "\n".join(found_params))
        if missing_params:
            st.error("Missing required parameters:\n" + "\n".join(missing_params))
            return None
        
        # Validate final parameters
        for param, value in params.items():
            if not isinstance(value, (int, float)):
                st.error(f"Invalid value for {param}: {value}")
                return None
        
        return params
        
    except Exception as e:
        st.error(f"Error in get_machine_params: {str(e)}")
        st.error(traceback.format_exc())
        return None
    
    except Exception as e:
        st.error(f"Error in get_machine_params: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None
        
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

    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")

    skip_torque_analysis = False
    is_machine_listed = st.radio(
        "Is your machine listed in the specifications file?",
        options=["Yes", "No"],
        index=0,
        help="Select 'No' if your machine type is not in the specifications file to proceed with basic analysis only."
    )

    if is_machine_listed == "No":
        skip_torque_analysis = True
        st.info("Proceeding with basic analysis only. Torque calculations will be skipped.")
    
    if not skip_torque_analysis and machine_specs_file is not None:
        try:
            machine_specs = load_machine_specs(machine_specs_file, 'xlsx')
            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)
            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved.")
                st.stop()

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
    elif not skip_torque_analysis:
        st.warning("Please upload Machine Specifications XLSX file.")
        return

    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

    if raw_data_file is not None:
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            sensor_columns = find_sensor_columns(df)
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
                df[revolution_col] = pd.to_numeric(df[revolution_col], errors='coerce')
                df[pressure_col] = pd.to_numeric(df[pressure_col], errors='coerce')
                df = df.dropna(subset=[pressure_col, revolution_col])

                time_col = st.selectbox(
                    "Select Time Column",
                    options=df.columns,
                    index=safe_get_loc(df.columns, sensor_columns.get('time', df.columns[0]))
                )
            
                df["Time_unit"] = handle_time_column(df, time_col)
                x_axis_max = update_plot_parameters(df, revolution_col)

                if not skip_torque_analysis:
                    n1 = machine_params.get("n1", df[revolution_col].max())
                    df = df[
                        (df[revolution_col] > 0.1) & 
                        (df[revolution_col] <= n1)
                    ]

                    def calculate_torque_wrapper(row):
                        working_pressure = row[pressure_col]
                        current_speed = row[revolution_col]
                        
                        if current_speed < 0.1:
                            return 0.0
                            
                        max_allowed_torque = machine_params["M_max_Vg1"]
                        
                        if current_speed < machine_params["n1"]:
                            torque = working_pressure * machine_params["torque_constant"]
                        else:
                            torque = (
                                (machine_params["n1"] / current_speed)
                                * machine_params["torque_constant"]
                                * working_pressure
                            )
                        
                        return round(min(torque, max_allowed_torque), 2)

                    df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)
                    torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
                    rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])
                    df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold

                    def M_max_Vg2(rpm):
                        return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

                    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
                    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])
                    rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)

                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(go.Scatter(
                        x=rpm_curve[rpm_curve <= elbow_rpm_cont],
                        y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_cont], machine_params['M_cont_value']),
                        mode='lines', name='M cont Max [kNm]', line=dict(color='green', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=rpm_curve[rpm_curve <= elbow_rpm_max],
                        y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']),
                        mode='lines', name='M max Vg1 [kNm]', line=dict(color='red', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=rpm_curve[rpm_curve <= machine_params['n1']],
                        y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]),
                        mode='lines', name='M max Vg2 [kNm]', line=dict(color='red', width=2, dash='dash')
                    ))

                    y_max_vg2 = M_max_Vg2(np.array([elbow_rpm_max, elbow_rpm_cont, machine_params['n1']]))
                    
                    fig.add_trace(go.Scatter(
                        x=[elbow_rpm_max, elbow_rpm_max], y=[0, y_max_vg2[0]],
                        mode='lines', line=dict(color='purple', width=1, dash='dot'), showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=[elbow_rpm_cont, elbow_rpm_cont], y=[0, y_max_vg2[1]],
                        mode='lines', line=dict(color='orange', width=1, dash='dot'), showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=[machine_params['n1'], machine_params['n1']], y=[0, y_max_vg2[2]],
                        mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False
                    ))

                    normal_data = df[~df['Is_Anomaly']]
                    anomaly_data = df[df['Is_Anomaly']]
                    torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers)]
                    rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

                    fig.add_trace(go.Scatter(
                        x=normal_data[revolution_col], y=normal_data['Calculated torque [kNm]'],
                        mode='markers', name='Normal Data',
                        marker=dict(color=normal_data['Calculated torque [kNm]'], colorscale='Viridis', size=8)
                    ))

                    fig.add_trace(go.Scatter(
                        x=anomaly_data[revolution_col], y=anomaly_data['Calculated torque [kNm]'],
                        mode='markers', name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)',
                        marker=dict(color='red', symbol='x', size=10)
                    ))

                    fig.add_trace(go.Scatter(
                        x=torque_outlier_data[revolution_col], y=torque_outlier_data['Calculated torque [kNm]'],
                        mode='markers', name='Torque Outliers',
                        marker=dict(color='orange', symbol='diamond', size=10)
                    ))

                    fig.add_trace(go.Scatter(
                        x=rpm_outlier_data[revolution_col], y=rpm_outlier_data['Calculated torque [kNm]'],
                        mode='markers', name='RPM Outliers',
                        marker=dict(color='purple', symbol='square', size=10)
                    ))

                    fig.add_hline(y=torque_upper_whisker, line_dash="dash", line_color="gray", annotation_text="Torque Upper Whisker")
                    fig.add_hline(y=torque_lower_whisker, line_dash="dot", line_color="gray", annotation_text="Torque Lower Whisker")

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
                    display_statistics(df, revolution_col, pressure_col)
                    display_explanation(anomaly_threshold)

                # Time-based visualization (always shown)
                st.subheader("Time-based Analysis")
                fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       subplot_titles=[revolution_col, pressure_col])
                
                fig_time.add_trace(
                    go.Scatter(x=df["Time_unit"], y=df[revolution_col], name=revolution_col),
                    row=1, col=1
                )
                
                fig_time.add_trace(
                    go.Scatter(x=df["Time_unit"], y=df[pressure_col], name=pressure_col),
                    row=2, col=1
                )

                fig_time.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig_time)

                # Download buttons
                st.sidebar.markdown("## Download Results")
                stats_df = pd.DataFrame({
                    'RPM': df[revolution_col].describe(),
                    'Working Pressure': df[pressure_col].describe()
                })
                if not skip_torque_analysis:
                    stats_df['Calculated Torque'] = df['Calculated torque [kNm]'].describe()
                
                st.sidebar.markdown(
                    get_table_download_link(stats_df, "statistical_analysis.csv", "Download Statistical Analysis"),
                    unsafe_allow_html=True
                )

    else:
        st.info("Please upload a Raw Data file to begin the analysis.")

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


def advanced_page():
    st.title("Advanced Analysis")
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications: XLSX (MM-Baureihenliste) or CSV format accepted", type=["xlsx", "csv"])

    skip_torque_analysis = False
    is_machine_listed = st.radio("Is your machine listed in the specifications file?", options=["Yes", "No"], index=0, 
                              help="Select 'No' if your machine type is not in the specifications file to proceed with basic analysis only.")

    if is_machine_listed == "No":
        skip_torque_analysis = True
        st.info("Proceeding with advanced analysis without torque calculations.")

    if not skip_torque_analysis and machine_specs_file is not None:
        try:
            file_type = machine_specs_file.name.split(".")[-1].lower()
            machine_specs = load_machine_specs(machine_specs_file, file_type)
            if machine_specs is None or machine_specs.empty:
                st.error("Machine specifications file is empty or could not be loaded.")
                st.stop()

            machine_types = machine_specs["Projekt"].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)
            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved.")
                st.stop()

            params_df = pd.DataFrame([machine_params])
            styled_table = params_df.style.set_table_styles([
                {"selector": "th", "props": [("border", "2px solid black"), ("padding", "5px")]},
                {"selector": "td", "props": [("border", "2px solid black"), ("padding", "5px")]},
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
        except Exception as e:
            st.error(f"An error occurred while processing the machine specifications: {str(e)}")
            st.stop()
    elif not skip_torque_analysis:
        st.warning("Please upload Machine Specifications file.")
        return

    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
    nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
    anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)
    num_cutting_rings = st.sidebar.number_input("Number of Cutting Rings", value=1, min_value=1, max_value=100)

    if raw_data_file is not None:
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            sensor_columns = find_sensor_columns(df)
            time_column_type_user = st.selectbox(
                "Select Time Column Type",
                options=["Numeric", "Datetime"],
                index=0,
                help="Choose how to interpret the Time column."
            )

            # Column selections
            pressure_col = st.selectbox("Select Pressure Column", options=df.columns,
                                      index=safe_get_loc(df.columns, sensor_columns.get('pressure', df.columns[0])))
            revolution_col = st.selectbox("Select Revolution Column", options=df.columns,
                                        index=safe_get_loc(df.columns, sensor_columns.get('revolution', df.columns[0])))
            advance_rate_col = st.selectbox("Select Advance Rate Column", options=df.columns,
                                          index=safe_get_loc(df.columns, sensor_columns.get('advance_rate', df.columns[0])))
            thrust_force_col = st.selectbox("Select Thrust Force Column", options=df.columns,
                                          index=safe_get_loc(df.columns, sensor_columns.get('thrust_force', df.columns[0])))
            distance_col = st.selectbox("Select Distance/Chainage Column", options=df.columns,
                                      index=safe_get_loc(df.columns, sensor_columns.get('distance', df.columns[0])))
            time_col = st.selectbox("Select Time Column", options=df.columns,
                                  index=safe_get_loc(df.columns, sensor_columns.get('time', df.columns[0])))

            # Data processing
            for col in [pressure_col, revolution_col, advance_rate_col, thrust_force_col, distance_col]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=[pressure_col, revolution_col, advance_rate_col, thrust_force_col, distance_col])

            # Time handling
            df[time_col] = df[time_col].fillna('')
            
            if time_column_type_user == "Numeric":
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                if df[time_col].isnull().all():
                    st.error(f"The selected time column '{time_col}' cannot be converted to numeric values.")
                    return
                time_column_type = 'numeric'
                df["Time_unit"] = df[time_col]
            else:
                try:
                    def extract_timestamp(x):
                        try:
                            if pd.isna(x) or x == '':
                                return None
                            timestamp_str = x.split('(')[0].strip()
                            return pd.to_datetime(timestamp_str)
                        except:
                            return None

                    df["Time_unit"] = df[time_col].apply(extract_timestamp)
                    
                    if df["Time_unit"].isnull().all():
                        df["Time_unit"] = pd.to_datetime(df[time_col], errors='coerce')
                        if df["Time_unit"].isnull().all():
                            st.error(f"Could not process time column '{time_col}'. Please check the format.")
                            return
                    
                    time_column_type = 'datetime'
                    
                except Exception as e:
                    st.error(f"Could not process time column '{time_col}'. Error: {str(e)}")
                    return

                if not success:
                    # Fall back to numeric if datetime conversion fails
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                    if df[time_col].isnull().all():
                        st.error(f"Could not process time column '{time_col}'. Please check the format.")
                        return
                    time_column_type = 'numeric'
                    df["Time_unit"] = df[time_col]

            # Sort and create range slider
            df = df.sort_values("Time_unit")
            if time_column_type == 'numeric':
                min_time_unit = df["Time_unit"].min()
                max_time_unit = df["Time_unit"].max()
                st.write(f"**Data time range:** {min_time_unit:.2f} to {max_time_unit:.2f} units")
                time_range = st.slider("Select Time Range", 
                                     min_value=float(min_time_unit), 
                                     max_value=float(max_time_unit),
                                     value=(float(min_time_unit), float(max_time_unit)), 
                                     format="%.2f")
            else:
                min_time_unit = df["Time_unit"].min().to_pydatetime()
                max_time_unit = df["Time_unit"].max().to_pydatetime()
                st.write(f"**Data time range:** {min_time_unit} to {max_time_unit}")
                time_step = timedelta(seconds=1)
                time_range = st.slider("Select Time Range", 
                                     min_value=min_time_unit, 
                                     max_value=max_time_unit,
                                     value=(min_time_unit, max_time_unit), 
                                     format="YYYY-MM-DD HH:mm:ss", 
                                     step=time_step)

            df = df[(df["Time_unit"] >= time_range[0]) & (df["Time_unit"] <= time_range[1])]

            # Calculate derived parameters
            df["Calculated Penetration Rate"] = df[advance_rate_col] / df[revolution_col]
            df["Thrust Force per Cutting Ring"] = df[thrust_force_col] / num_cutting_rings

            if not skip_torque_analysis:
                n1 = machine_params.get("n1", df[revolution_col].max())
                df = df[(df[revolution_col] > 0.1) & (df[revolution_col] <= n1)]

                def calculate_torque_wrapper(row):
                    working_pressure = row[pressure_col]
                    current_speed = row[revolution_col]
                    
                    if current_speed < 0.1:
                        return 0.0
                        
                    max_allowed_torque = machine_params["M_max_Vg1"]
                    
                    if current_speed < machine_params["n1"]:
                        torque = working_pressure * machine_params["torque_constant"]
                    else:
                        torque = ((machine_params["n1"] / current_speed) * machine_params["torque_constant"] * working_pressure)
                    
                    return round(min(torque, max_allowed_torque), 2)

                df['Calculated torque [kNm]'] = df.apply(calculate_torque_wrapper, axis=1)

                # Torque visualization
                x_axis_max = machine_params["n1"] * 1.2
                torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
                rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])
                df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold

                def M_max_Vg2(rpm):
                    return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

                elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
                elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])
                rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)

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
                                      mode='lines', name='M max Vg2 [kNm]',
                                      line=dict(color='red', width=2, dash='dash')))

                y_max_vg2 = M_max_Vg2(np.array([elbow_rpm_max, elbow_rpm_cont, machine_params['n1']]))
                
                # Add vertical lines
                fig.add_trace(go.Scatter(x=[elbow_rpm_max, elbow_rpm_max], y=[0, y_max_vg2[0]],
                                      mode='lines', line=dict(color='purple', width=1, dash='dot'),
                                      showlegend=False))

                fig.add_trace(go.Scatter(x=[elbow_rpm_cont, elbow_rpm_cont], y=[0, y_max_vg2[1]],
                                      mode='lines', line=dict(color='orange', width=1, dash='dot'),
                                      showlegend=False))

                fig.add_trace(go.Scatter(x=[machine_params['n1'], machine_params['n1']], y=[0, y_max_vg2[2]],
                                      mode='lines', line=dict(color='black', width=1, dash='dash'),
                                      showlegend=False))

                # Plot data points
                normal_data = df[~df['Is_Anomaly']]
                anomaly_data = df[df['Is_Anomaly']]
                torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers)]
                rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

                fig.add_trace(go.Scatter(x=normal_data[revolution_col], y=normal_data['Calculated torque [kNm]'],
                                      mode='markers', name='Normal Data',
                                      marker=dict(color=normal_data['Calculated torque [kNm]'],
                                                colorscale='Viridis', size=8)))

                fig.add_trace(go.Scatter(x=anomaly_data[revolution_col], y=anomaly_data['Calculated torque [kNm]'],
                                      mode='markers', name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)',
                                      marker=dict(color='red', symbol='x', size=10)))

                fig.add_trace(go.Scatter(x=torque_outlier_data[revolution_col],
                                      y=torque_outlier_data['Calculated torque [kNm]'],
                                      mode='markers', name='Torque Outliers',
                                      marker=dict(color='orange', symbol='diamond', size=10)))

                fig.add_trace(go.Scatter(x=rpm_outlier_data[revolution_col], y=rpm_outlier_data['Calculated torque [kNm]'],
                                      mode='markers', name='RPM Outliers',
                                      marker=dict(color='purple', symbol='square', size=10)))

                fig.add_hline(y=torque_upper_whisker, line_dash="dash", line_color="gray",
                           annotation_text="Torque Upper Whisker")
                fig.add_hline(y=torque_lower_whisker, line_dash="dot", line_color="gray", 
                           annotation_text="Torque Lower Whisker")

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
                display_statistics(df, revolution_col, pressure_col, thrust_force_col)

            # Features visualization
            window_size = st.sidebar.slider("Select Rolling Window Size", min_value=10, max_value=1000, value=100, step=10)
            show_means = st.checkbox("Show Mean Values", value=True)

            features = [
                {"column": advance_rate_col, "display_name": "Advance Rate", "color": "blue"},
                {"column": "Calculated Penetration Rate", "display_name": "Penetration Rate", "color": "green"},
                {"column": thrust_force_col, "display_name": "Thrust Force", "color": "red"},
                {"column": "Thrust Force per Cutting Ring", "display_name": "Thrust Force per Cutting Ring", "color": "orange"},
                {"column": revolution_col, "display_name": "Revolution", "color": "purple"},
                {"column": pressure_col, "display_name": "Working Pressure", "color": "cyan"}
            ]

            if not skip_torque_analysis:
                features.append({
                    "column": "Calculated torque [kNm]",
                    "display_name": "Calculated Torque [kNm]",
                    "color": "magenta"
                })

            num_features = len(features)

            for feature in features:
                df[f"{feature['column']}_mean"] = df[feature['column']].rolling(window=window_size, min_periods=1).mean()

            fig_time = make_subplots(
                rows=2*num_features,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=None
            )

            for i, feature in enumerate(features, start=1):
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
                fig_time.update_yaxes(title_text=feature["display_name"], row=2*i-1, col=1)

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
                    fig_time.update_yaxes(title_text=f"{feature['display_name']} - Rolling Mean", row=2*i, col=1)

            fig_time.update_layout(
                xaxis_title="Time",
                height=300 * 2 * num_features,
                showlegend=False,
                title_text="Features over Time (Original and Rolling Mean)",
            )

            st.plotly_chart(fig_time, use_container_width=True)

            window_size_distance = st.sidebar.slider(
                "Select Rolling Window Size (Distance)",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            show_means_distance = st.checkbox("Show Rolling Mean Values (Distance)", value=True)

            df = df.sort_values(by=distance_col)

            for feature in features:
                if feature['column'] in df.columns:
                    df[f"{feature['column']}_distance_mean"] = df[feature['column']].rolling(
                        window=window_size_distance, min_periods=1
                    ).mean()

            fig_distance = make_subplots(
                rows=2*num_features,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=None
            )

            for i, feature in enumerate(features, start=1):
                if feature['column'] in df.columns:
                    fig_distance.add_trace(
                        go.Scatter(
                            x=df[distance_col],
                            y=df[feature["column"]],
                            mode="lines",
                            name=feature["display_name"],
                            line=dict(color=feature["color"]),
                        ),
                        row=2*i-1,
                        col=1,
                    )
                    fig_distance.update_yaxes(title_text=feature["display_name"], row=2*i-1, col=1)

                    if show_means_distance:
                        fig_distance.add_trace(
                            go.Scatter(
                                x=df[distance_col],
                                y=df[f"{feature['column']}_distance_mean"],
                                mode="lines",
                                name=f"{feature['display_name']} Rolling Mean",
                                line=dict(color=feature["color"], dash="dash"),
                            ),
                            row=2*i,
                            col=1,
                        )
                        fig_distance.update_yaxes(title_text=f"{feature['display_name']} Rolling Mean", row=2*i, col=1)

            fig_distance.update_layout(
                xaxis_title="Distance/Chainage",
                height=300 * 2 * num_features,
                showlegend=False,
                title_text="Features over Distance/Chainage (Original and Rolling Mean)",
            )

            st.plotly_chart(fig_distance, use_container_width=True)

            st.write("""
            **Interpretation Guide:**
            - **Advance Rate**: Speed of machine advancement. Fluctuations may indicate changing ground conditions.
            - **Penetration Rate**: Advance Rate per Revolution. Shows material penetration efficiency.
            - **Thrust Force**: Force at cutting head. High values may indicate hard ground or mechanical issues.
            - **Thrust Force per Cutting Ring**: Load per ring indicating distribution.
            - **Revolution**: Cutting head rotational speed. Affects penetration rate and torque.
            - **Working Pressure**: Operating pressure. Changes may indicate anomalies or adjustments.
            """)

            st.sidebar.markdown("## Download Results")
            stats_df = pd.DataFrame({
                "RPM": df[revolution_col].describe(),
                "Working Pressure": df[pressure_col].describe(),
                "Advance Rate": df[advance_rate_col].describe(),
                "Penetration Rate": df["Calculated Penetration Rate"].describe(),
                "Thrust Force": df[thrust_force_col].describe(),
                "Thrust Force per Cutting Ring": df["Thrust Force per Cutting Ring"].describe()
            })
            
            if not skip_torque_analysis:
                stats_df["Calculated Torque"] = df["Calculated torque [kNm]"].describe()
            
            st.sidebar.markdown(
                get_table_download_link(stats_df, "advanced_statistical_analysis.csv", "Download Statistical Analysis"),
                unsafe_allow_html=True
            )

        else:
            st.info("Please upload a Raw Data file to begin the analysis.")


if __name__ == "__main__":
    main()
