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
import re
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# ----------------------------
# Configure Logging
# ----------------------------
def setup_logging():
    """Configure logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('torquevision.log')
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# ----------------------------
# Custom Exceptions
# ----------------------------
class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

# ----------------------------
# Error Handling Function
# ----------------------------
def handle_error(error, error_type="Error"):
    """
    Unified error handling.
    
    Args:
        error: The error that occurred
        error_type (str): Type of error for display
    """
    error_msg = str(error)
    logger.error(f"{error_type}: {error_msg}")
    
    if isinstance(error, DataProcessingError):
        st.error(f"Data Processing Error: {error_msg}")
    elif isinstance(error, ValueError):
        st.error(f"Value Error: {error_msg}")
    else:
        st.error(f"An error occurred: {error_msg}")
    
    # Log additional debug information if available
    if hasattr(error, '__cause__') and error.__cause__:
        logger.error(f"Caused by: {str(error.__cause__)}")

# ----------------------------
# DataLoader Class
# ----------------------------
class DataLoader:
    def __init__(self):
        self.sensor_patterns = {
            "pressure": [
                r"(?i).*pressure.*",
                r"(?i).*press.*",
                r"(?i).*bar.*",
                r"(?i).*druck.*",
                r"(?i).*pression.*",
                "Working pressure [bar]",
                "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26",
                "Pression [bar]",
                "Presión [bar]",
                "Pressure",
                "cutting wheel.MPU1WPr",
                "MPU1WPr"
            ],
            "revolution": [
                r"(?i).*revolution.*",
                r"(?i).*rev.*",
                r"(?i).*rpm.*",
                r"(?i).*speed.*",
                r"(?i).*drehzahl.*",
                "Revolution [rpm]",
                "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30",
                "Vitesse [rpm]",
                "Revoluciones [rpm]",
                "RPM",
                "Speed",
                "cutting wheel.CWSpeed",
                "CWSpeed"
            ],
            "time": [
                r"(?i).*time.*",
                r"(?i).*zeit.*",
                r"(?i).*date.*",
                r"(?i).*datum.*",
                r"(?i).*timestamp.*",
                "Time",
                "Timestamp",
                "DateTime",
                "Date",
                "Zeit",
                "Relativzeit",
                "Uhrzeit",
                "ts(utc)"
            ],
            "advance_rate": [
                r"(?i).*advance.*rate.*",
                r"(?i).*speed.*",
                r"(?i).*velocity.*",
                r"(?i).*geschw.*",
                "Advance Rate",
                "Vorschubgeschwindigkeit",
                "Avance",
                "Rate of Penetration",
                "ROP",
                "Advance [m/min]",
                "VTgeschw_Z",
                "VTgeschw"
            ],
            "thrust_force": [
                r"(?i).*thrust.*force.*",
                r"(?i).*thrust.*",
                r"(?i).*force.*",
                r"(?i).*kraft.*",
                "Thrust Force",
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
                r"(?i).*distance.*",
                r"(?i).*chainage.*",
                r"(?i).*position.*",
                r"(?i).*station.*",
                r"(?i).*weg.*",
                "Distance",
                "Chainage",
                "Position",
                "Kette",
                "Station",
                "V34_TL_SR_m_Z",
                "TL_SR_m_Z",
                "SR_m_Z",
                "Weg"
            ]
        }

    def detect_encoding(self, file_content: bytes) -> str:
        detected = chardet.detect(file_content)
        encoding = detected['encoding']
        
        if not encoding or detected['confidence'] < 0.8:
            for enc in ['utf-8', 'iso-8859-1', 'latin1', 'cp1252']:
                try:
                    file_content.decode(enc)
                    return enc
                except UnicodeDecodeError:
                    continue
        return encoding

    def detect_delimiter(self, content: str) -> str:
        try:
            dialect = csv.Sniffer().sniff(content[:4096])
            return dialect.delimiter
        except:
            delimiters = [',', ';', '\t', '|']
            max_cols = 0
            best_delimiter = ','
            
            for delimiter in delimiters:
                try:
                    df_test = pd.read_csv(StringIO(content), sep=delimiter, nrows=5)
                    num_cols = len(df_test.columns)
                    if num_cols > max_cols:
                        max_cols = num_cols
                        best_delimiter = delimiter
                except:
                    continue
            return best_delimiter

    def find_matching_column(self, columns: List[str], sensor_type: str) -> Optional[str]:
        patterns = self.sensor_patterns[sensor_type]
        
        # First try exact matches
        for pattern in patterns:
            if not pattern.startswith('(?i)'):  # Only check exact patterns
                if pattern in columns:
                    return pattern
        
        # Then try regex patterns
        for pattern in patterns:
            if pattern.startswith('(?i)'):
                for column in columns:
                    if re.search(pattern, column, re.IGNORECASE):
                        return column
        
        return None

    def load_data(self, file, file_type: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        try:
            if file_type == 'csv':
                file_content = file.read()
                encoding = self.detect_encoding(file_content)
                content_str = file_content.decode(encoding)
                delimiter = self.detect_delimiter(content_str)
                
                df = pd.read_csv(
                    StringIO(content_str),
                    sep=delimiter,
                    encoding=encoding,
                    on_bad_lines='warn',
                    low_memory=False
                )
            else:  # Excel
                df = pd.read_excel(file, engine='openpyxl')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Find matching columns for each sensor type
            column_mapping = {}
            for sensor_type in self.sensor_patterns.keys():
                matched_column = self.find_matching_column(df.columns.tolist(), sensor_type)
                if matched_column:
                    column_mapping[sensor_type] = matched_column
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if df[col].dtype != 'object']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df, column_mapping
        except Exception as e:
            logger.exception("Error loading data")
            raise DataProcessingError(f"Failed to load data: {str(e)}")

# ----------------------------
# Utility Functions
# ----------------------------
def get_table_download_link(df: pd.DataFrame, filename: str, text: str) -> str:
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    fractional_seconds = td.microseconds / 1_000_000

    formatted_time = ""
    if days > 0:
        formatted_time += f"{days} day{'s' if days != 1 else ''}, "
    if hours > 0:
        formatted_time += f"{hours} hour{'s' if hours != 1 else ''}, "
    if minutes > 0:
        formatted_time += f"{minutes} minute{'s' if minutes != 1 else ''}, "

    total_sec = seconds + fractional_seconds
    formatted_time += f"{total_sec:.2f} second{'s' if total_sec != 1 else ''}"

    return formatted_time

def calculate_whisker_and_outliers(data: pd.Series) -> Tuple[float, float, pd.Series]:
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    whisker_length = 1.5 * IQR
    lower_whisker = Q1 - whisker_length
    upper_whisker = Q3 + whisker_length
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    return lower_whisker, upper_whisker, outliers

def calculate_whisker_and_outliers_advanced(data: pd.Series) -> Tuple[float, float, pd.Series]:
    Q1 = data.quantile(0.10)
    Q3 = data.quantile(0.90)
    IQR = Q3 - Q1
    whisker_length = 1.5 * IQR
    lower_whisker = Q1 - whisker_length
    upper_whisker = Q3 + whisker_length
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    return lower_whisker, upper_whisker, outliers

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link for DataFrame."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{filename}</a>'
    return href

# ----------------------------
# Plotting Functions
# ----------------------------
def calculate_torque_wrapper(row, pressure_col, revolution_col, machine_params):
    working_pressure = row[pressure_col]
    current_speed = row[revolution_col]

    if current_speed < machine_params['n1']:
        torque = working_pressure * machine_params['torque_constant']
    else:
        torque = (machine_params['n1'] / current_speed) * machine_params['torque_constant'] * working_pressure

    return round(torque, 2)

def create_torque_visualization(df, revolution_col, pressure_col, machine_params, P_max, nu, anomaly_threshold):
    # Calculate whiskers and outliers
    torque_lower_whisker, torque_upper_whisker, torque_outliers = calculate_whisker_and_outliers(df['Calculated torque [kNm]'])
    rpm_lower_whisker, rpm_upper_whisker, rpm_outliers = calculate_whisker_and_outliers(df[revolution_col])

    # Anomaly detection
    df['Is_Anomaly'] = df[pressure_col] >= anomaly_threshold

    # Calculate M max Vg2
    def M_max_Vg2(rpm):
        return np.minimum(machine_params['M_max_Vg1'], (P_max * 60 * nu) / (2 * np.pi * rpm))

    # Calculate elbow points
    elbow_rpm_max = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_max_Vg1'])
    elbow_rpm_cont = (P_max * 60 * nu) / (2 * np.pi * machine_params['M_cont_value'])

    # Generate RPM values for curves
    rpm_curve = np.linspace(0.1, machine_params['n1'], 1000)

    # Create figure
    fig = make_subplots(rows=1, cols=1)

    # Add torque curves
    fig.add_trace(go.Scatter(
        x=rpm_curve[rpm_curve <= elbow_rpm_max],
        y=np.full_like(rpm_curve[rpm_curve <= elbow_rpm_max], machine_params['M_max_Vg1']),
        mode='lines',
        name='M max Vg1 [kNm]',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=rpm_curve[rpm_curve <= machine_params['n1']],
        y=M_max_Vg2(rpm_curve[rpm_curve <= machine_params['n1']]),
        mode='lines',
        name='M max Vg2 [kNm]',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Calculate y-values for vertical lines
    y_max_vg2 = M_max_Vg2(np.array([elbow_rpm_max, elbow_rpm_cont, machine_params['n1']]))

    # Add vertical lines
    fig.add_trace(go.Scatter(
        x=[elbow_rpm_max, elbow_rpm_max],
        y=[0, y_max_vg2[0]],
        mode='lines',
        line=dict(color='purple', width=1, dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[elbow_rpm_cont, elbow_rpm_cont],
        y=[0, y_max_vg2[1]],
        mode='lines',
        line=dict(color='orange', width=1, dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[machine_params['n1'], machine_params['n1']],
        y=[0, y_max_vg2[2]],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False
    ))

    # Separate data points
    normal_data = df[~df['Is_Anomaly']]
    anomaly_data = df[df['Is_Anomaly']]
    torque_outlier_data = df[df['Calculated torque [kNm]'].isin(torque_outliers)]
    rpm_outlier_data = df[df[revolution_col].isin(rpm_outliers)]

    # Add data points
    fig.add_trace(go.Scatter(
        x=normal_data[revolution_col],
        y=normal_data['Calculated torque [kNm]'],
        mode='markers',
        name='Normal Data',
        marker=dict(
            color=normal_data['Calculated torque [kNm]'],
            colorscale='Viridis',
            size=8
        )
    ))

    fig.add_trace(go.Scatter(
        x=anomaly_data[revolution_col],
        y=anomaly_data['Calculated torque [kNm]'],
        mode='markers',
        name=f'Anomaly (Pressure ≥ {anomaly_threshold} bar)',
        marker=dict(color='red', symbol='x', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=torque_outlier_data[revolution_col],
        y=torque_outlier_data['Calculated torque [kNm]'],
        mode='markers',
        name='Torque Outliers',
        marker=dict(color='orange', symbol='diamond', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=rpm_outlier_data[revolution_col],
        y=rpm_outlier_data['Calculated torque [kNm]'],
        mode='markers',
        name='RPM Outliers',
        marker=dict(color='purple', symbol='square', size=10)
    ))

    # Add whisker lines
    fig.add_hline(
        y=torque_upper_whisker,
        line_dash="dash",
        line_color="gray",
        annotation_text="Torque Upper Whisker"
    )
    fig.add_hline(
        y=torque_lower_whisker,
        line_dash="dot",
        line_color="gray",
        annotation_text="Torque Lower Whisker"
    )

    # Update layout
    fig.update_layout(
        title=f'Torque Analysis',
        xaxis_title='Revolution [1/min]',
        yaxis_title='Torque [kNm]',
        xaxis=dict(range=[0, df[revolution_col].max() * 1.1]),
        yaxis=dict(range=[0, max(60, df['Calculated torque [kNm]'].max() * 1.1)]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        height=800,
        margin=dict(l=50, r=50, t=100, b=100)
    )

    return fig

def create_time_series_visualization(df, columns):
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=('Pressure', 'Revolution', 'Torque', 'Thrust Force'),
        vertical_spacing=0.08
    )

    # Pressure trace
    fig.add_trace(
        go.Scatter(
            x=df[columns['time']],
            y=df[columns['pressure']],
            name='Pressure'
        ),
        row=1, col=1
    )

    # Revolution trace
    fig.add_trace(
        go.Scatter(
            x=df[columns['time']],
            y=df[columns['revolution']],
            name='Revolution'
        ),
        row=2, col=1
    )

    # Torque trace
    fig.add_trace(
        go.Scatter(
            x=df[columns['time']],
            y=df['Calculated torque [kNm]'],
            name='Torque'
        ),
        row=3, col=1
    )

    # Thrust force trace
    fig.add_trace(
        go.Scatter(
            x=df[columns['time']],
            y=df[columns['thrust_force']],
            name='Thrust Force'
        ),
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Time Series Analysis"
    )

    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Pressure [bar]", row=1, col=1)
    fig.update_yaxes(title_text="Revolution [rpm]", row=2, col=1)
    fig.update_yaxes(title_text="Torque [kNm]", row=3, col=1)
    fig.update_yaxes(title_text="Thrust Force [kN]", row=4, col=1)

    return fig

def create_distance_visualization(df, columns):
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=('Pressure', 'Revolution', 'Torque', 'Advance Rate'),
        vertical_spacing=0.08
    )

    # Pressure vs. Distance
    fig.add_trace(
        go.Scatter(
            x=df[columns['distance']],
            y=df[columns['pressure']],
            name='Pressure'
        ),
        row=1, col=1
    )

    # Revolution vs. Distance
    fig.add_trace(
        go.Scatter(
            x=df[columns['distance']],
            y=df[columns['revolution']],
            name='Revolution'
        ),
        row=2, col=1
    )

    # Torque vs. Distance
    fig.add_trace(
        go.Scatter(
            x=df[columns['distance']],
            y=df['Calculated torque [kNm]'],
            name='Torque'
        ),
        row=3, col=1
    )

    # Advance Rate vs. Distance
    fig.add_trace(
        go.Scatter(
            x=df[columns['distance']],
            y=df[columns['advance_rate']],
            name='Advance Rate'
        ),
        row=4, col=1
    )

    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Distance Analysis"
    )

    fig.update_xaxes(title_text="Distance [m]", row=4, col=1)
    fig.update_yaxes(title_text="Pressure [bar]", row=1, col=1)
    fig.update_yaxes(title_text="Revolution [rpm]", row=2, col=1)
    fig.update_yaxes(title_text="Torque [kNm]", row=3, col=1)
    fig.update_yaxes(title_text="Advance Rate [mm/min]", row=4, col=1)

    return fig

# ----------------------------
# Machine Parameters Functions
# ----------------------------
def get_machine_params(specs_df: pd.DataFrame, machine_type: str) -> Optional[Dict[str, float]]:
    """
    Get machine parameters with enhanced error handling and validation.
    
    Args:
        specs_df (pd.DataFrame): DataFrame containing machine specifications
        machine_type (str): Selected machine type
        
    Returns:
        dict: Machine parameters
    """
    try:
        # Filter for selected machine type
        machine_rows = specs_df[specs_df['Projekt'] == machine_type]
        if machine_rows.empty:
            raise DataProcessingError(f"Machine type '{machine_type}' not found in specifications")

        # Extract first matching row
        machine_data = machine_rows.iloc[0]

        # Define parameter mappings with alternatives and German translations
        parameter_mappings = {
            'n1': [
                'n1[1/min]', 'n1 (1/min)', 'n1[rpm]', 'Max RPM', 'n1', 
                'Maximum Speed', 'Max Speed', 'Drehzahl Max',
                'MaxSpeed', 'Speed_Max', 'n_max'
            ],
            'n2': [
                'n2[1/min]', 'n2 (1/min)', 'n2[rpm]', 'Min RPM', 'n2',
                'Minimum Speed', 'Min Speed', 'Drehzahl Min',
                'MinSpeed', 'Speed_Min', 'n_min'
            ],
            'M_cont_value': [
                'M(dauer) [kNm]', 'M(dauer)[kNm]', 'M (dauer)', 'Continuous Torque',
                'M_cont', 'Cont_Torque', 'Dauer_Drehmoment', 'M_dauer',
                'Continuous_Torque', 'TorqueContinuous', 'M cont'
            ],
            'M_max_Vg1': [
                'M(max)', 'M max', 'M (max)', 'M_max[kNm]', 'M(max)[kNm]', 
                'Max Torque', 'Maximum_Torque', 'TorqueMax',
                'Max_Torque', 'Drehmoment_Max', 'M maximum'
            ],
            'torque_constant': [
                'Drehmomentumrechnung[kNm/bar]', 'Drehmomentumrechnung [kNm/bar]',
                'Torque Constant', 'TorqueConstant', 'Torque_Constant',
                'Drehmoment_Konstante', 'Torque_Factor', 'TorqueFactor',
                'kNm/bar', 'Torque/Pressure', 'TorquePerPressure'
            ]
        }

        # Add case-insensitive search
        for param_name, possible_names in parameter_mappings.items():
            # Add lowercase and uppercase versions
            extra_names = []
            for name in possible_names:
                extra_names.extend([name.lower(), name.upper()])
            parameter_mappings[param_name].extend(extra_names)

        # Find parameters
        params = {}
        missing_params = []
        
        for param_name, possible_names in parameter_mappings.items():
            found = False
            # First try exact matches
            for name in possible_names:
                if name in machine_data.index:
                    try:
                        value = machine_data[name]
                        # Handle string values that might contain numbers
                        if isinstance(value, str):
                            # Remove any units in brackets and convert to float
                            value = float(re.sub(r'\[.*?\]', '', value).strip())
                        params[param_name] = float(value)
                        found = True
                        break
                    except (ValueError, TypeError):
                        continue
            
            # If no exact match, try fuzzy matching
            if not found:
                for column in machine_data.index:
                    if any(possible_name.lower() in column.lower() for possible_name in possible_names):
                        try:
                            value = machine_data[column]
                            if isinstance(value, str):
                                value = float(re.sub(r'\[.*?\]', '', value).strip())
                            params[param_name] = float(value)
                            found = True
                            break
                        except (ValueError, TypeError):
                            continue
            
            if not found:
                missing_params.append(f"{param_name} ({possible_names[0]})")

        if missing_params:
            # If parameters are missing, try to provide more helpful error message
            available_columns = ", ".join(machine_data.index)
            raise DataProcessingError(
                f"Missing required parameters for machine '{machine_type}': {', '.join(missing_params)}\n"
                f"Available columns: {available_columns}"
            )

        # Add default values for missing parameters if appropriate
        if 'n2' not in params and 'n1' in params:
            params['n2'] = params['n1'] * 0.1  # Default minimum speed to 10% of max speed
        
        if 'M_cont_value' not in params and 'M_max_Vg1' in params:
            params['M_cont_value'] = params['M_max_Vg1'] * 0.8  # Default continuous torque to 80% of max

        # Validate parameter values
        validate_machine_params(params)

        return params

    except DataProcessingError as e:
        logger.error(f"DataProcessingError: {str(e)}")
        st.error(f"DataProcessingError: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_machine_params: {str(e)}")
        st.error(f"Unexpected error in get_machine_params: {str(e)}")
        return None

def validate_machine_params(params: Dict[str, float]):
    """
    Validate machine parameters.
    
    Args:
        params (dict): Machine parameters to validate
        
    Raises:
        DataProcessingError: If validation fails
    """
    # Check for positive values
    for param, value in params.items():
        if value <= 0:
            raise DataProcessingError(f"Parameter {param} must be positive, got {value}")

    # Check logical relationships
    if params['n2'] >= params['n1']:
        raise DataProcessingError(
            f"Minimum RPM (n2={params['n2']}) must be less than maximum RPM (n1={params['n1']})"
        )

    if params['M_cont_value'] >= params['M_max_Vg1']:
        raise DataProcessingError(
            f"Continuous torque ({params['M_cont_value']}) must be less than maximum torque ({params['M_max_Vg1']})"
        )

# ----------------------------
# Data Processing Functions
# ----------------------------
def validate_data(df: pd.DataFrame, columns: Dict[str, str]):
    """
    Validate loaded data.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        columns (dict): Selected column mappings
        
    Raises:
        DataProcessingError: If validation fails
    """
    # Check for minimum required data points
    if len(df) < 10:
        raise DataProcessingError("Insufficient data points (minimum 10 required)")

    # Check for duplicate columns
    if len(set(columns.values())) != len(columns):
        raise DataProcessingError("Duplicate columns selected")

    # Check numeric columns
    numeric_cols = ['pressure', 'revolution', 'advance_rate', 'thrust_force']
    for col_type in numeric_cols:
        if col_type in columns:
            col = columns[col_type]
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataProcessingError(
                    f"Non-numeric values found in {col_type} column: {col}"
                )

def safe_numeric_conversion(series: pd.Series, col_name: str) -> pd.Series:
    """
    Safely convert series to numeric with warnings.
    
    Args:
        series (pd.Series): Series to convert
        col_name (str): Column name for error messages
        
    Returns:
        pd.Series: Converted numeric series
    """
    try:
        numeric_series = pd.to_numeric(series, errors='coerce')
        null_count = numeric_series.isnull().sum()
        if null_count > 0:
            st.warning(
                f"{null_count} values in column '{col_name}' could not be converted to numeric "
                f"({null_count/len(series)*100:.1f}% of data)"
            )
        return numeric_series
    except Exception as e:
        raise DataProcessingError(f"Error converting column '{col_name}' to numeric: {str(e)}")

def process_time_column(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Process time column with various format support.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        time_col (str): Name of time column
        
    Returns:
        pd.Series: Processed time column
    """
    try:
        # First try datetime conversion
        try:
            time_series = pd.to_datetime(df[time_col])
            return time_series
        except:
            pass

        # Try numeric conversion (assuming seconds or similar)
        try:
            numeric_time = pd.to_numeric(df[time_col])
            # If values are too large, might be Unix timestamps
            if numeric_time.min() > 1e9:
                return pd.to_datetime(numeric_time, unit='s')
            return numeric_time
        except:
            pass

        raise DataProcessingError(
            f"Could not process time column '{time_col}'. "
            "Please ensure it contains datetime or numeric values."
        )

    except Exception as e:
        logger.error(f"Error processing time column: {str(e)}")
        st.error(f"Error processing time column: {str(e)}")
        return None

def process_advanced_data(df: pd.DataFrame, columns: Dict[str, str], num_cutting_rings: int, machine_params: Dict[str, float]) -> pd.DataFrame:
    """
    Process data for advanced analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with raw data
        columns (dict): Column mappings
        num_cutting_rings (int): Number of cutting rings
        machine_params (dict): Machine parameters
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Calculate Penetration Rate
    df['Calculated Penetration Rate'] = df[columns['advance_rate']] / df[columns['revolution']]
    
    # Calculate Thrust Force per Cutting Ring
    df['Thrust Force per Cutting Ring'] = df[columns['thrust_force']] / num_cutting_rings
    
    # Calculate torque
    df['Calculated torque [kNm]'] = df.apply(
        lambda row: calculate_torque_wrapper(
            row, 
            columns['pressure'],
            columns['revolution'],
            machine_params
        ),
        axis=1
    )
    
    return df

def create_statistics_df(df: pd.DataFrame, revolution_col: str, pressure_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        'RPM': df[revolution_col].describe(),
        'Calculated_Torque': df['Calculated torque [kNm]'].describe(),
        'Working_Pressure': df[pressure_col].describe()
    })

def create_advanced_statistics_df(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        'Pressure': df[columns['pressure']].describe(),
        'Revolution': df[columns['revolution']].describe(),
        'Torque': df['Calculated torque [kNm]'].describe(),
        'Advance_Rate': df[columns['advance_rate']].describe(),
        'Penetration_Rate': df['Calculated Penetration Rate'].describe(),
        'Thrust_Force': df[columns['thrust_force']].describe(),
        'Thrust_Force_per_Ring': df['Thrust Force per Cutting Ring'].describe()
    })

def display_statistical_summary(df: pd.DataFrame, revolution_col: str, pressure_col: str, thrust_force_col: Optional[str] = None):
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

def display_advanced_statistical_summary(df: pd.DataFrame, columns: Dict[str, str]):
    st.subheader("Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Basic Measurements:")
        st.write(pd.DataFrame({
            'Pressure': df[columns['pressure']].describe(),
            'Revolution': df[columns['revolution']].describe(),
            'Torque': df['Calculated torque [kNm]'].describe()
        }))
    
    with col2:
        st.write("Advanced Measurements:")
        st.write(pd.DataFrame({
            'Advance_Rate': df[columns['advance_rate']].describe(),
            'Penetration_Rate': df['Calculated Penetration Rate'].describe(),
            'Thrust_Force': df[columns['thrust_force']].describe(),
            'Thrust_Force_per_Ring': df['Thrust Force per Cutting Ring'].describe()
        }))

def display_explanation(anomaly_threshold: float):
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
    """)

def display_data_info(df: pd.DataFrame, columns: Dict[str, str]):
    """
    Display comprehensive data information.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        columns (dict): Selected column mappings
    """
    st.subheader("Data Overview")
    
    # Basic info
    st.write(f"Total Records: {len(df):,}")
    st.write(f"Time Range: {df[columns['time']].min()} to {df[columns['time']].max()}")
    
    # Data quality
    null_counts = df[list(columns.values())].isnull().sum()
    if null_counts.any():
        st.warning("Missing Values Detected:")
        for col, count in null_counts.items():
            if count > 0:
                st.write(f"- {col}: {count:,} missing values ({count/len(df)*100:.1f}%)")

    # Value ranges
    st.write("### Value Ranges")
    for col_type, col_name in columns.items():
        if col_type in ['pressure', 'revolution', 'advance_rate', 'thrust_force']:
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            st.write(f"{col_type.title()}: {min_val:.2f} to {max_val:.2f}")

def calculate_performance_metrics(df: pd.DataFrame, columns: Dict[str, str]) -> Dict[str, Union[timedelta, float]]:
    """
    Calculate additional performance metrics.
    
    Args:
        df (pd.DataFrame): DataFrame with measurements
        columns (dict): Column mappings
        
    Returns:
        dict: Calculated metrics
    """
    metrics = {}
    
    # Operating time
    time_range = df[columns['time']].max() - df[columns['time']].min()
    metrics['operating_time'] = time_range
    
    # Average values
    metrics['avg_pressure'] = df[columns['pressure']].mean()
    metrics['avg_revolution'] = df[columns['revolution']].mean()
    metrics['avg_torque'] = df['Calculated torque [kNm]'].mean()
    
    # Performance indicators
    if 'advance_rate' in columns:
        metrics['total_advance'] = (
            df[columns['advance_rate']].mean() * (time_range.total_seconds() / 3600)
        )
    
    # Efficiency metrics
    metrics['pressure_utilization'] = (
        df[columns['pressure']].mean() / df[columns['pressure']].max()
    )
    
    return metrics

# ----------------------------
# Main Application Pages
# ----------------------------
def original_page():
    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")

    data_loader = DataLoader()
    
    # File uploaders for batch data
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type=["xlsx"])

    # Load machine specs if available
    if machine_specs_file is not None:
        try:
            machine_specs = pd.read_excel(machine_specs_file)
            if 'Projekt' not in machine_specs.columns:
                st.error("Machine specifications file must contain a 'Projekt' column.")
                st.stop()

            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

            # Get machine parameters
            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved.")
                st.stop()

            # Display machine parameters
            params_df = pd.DataFrame([machine_params])
            st.markdown(
                """
                <style>
                table {
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 0.9em;
                    font-family: sans-serif;
                    min-width: 400px;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                }
                table thead tr {
                    background-color: rgb(0, 62, 37);
                    color: #ffffff;
                    text-align: left;
                }
                table th,
                table td {
                    padding: 12px 15px;
                    border: 2px solid black;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.table(params_df)

            # Sidebar for user inputs
            st.sidebar.header("Parameter Settings")
            P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
            nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
            anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)

            if raw_data_file is not None:
                # Load and process data
                file_type = raw_data_file.name.split(".")[-1].lower()
                df, detected_columns = data_loader.load_data(raw_data_file, file_type)

                if df is not None:
                    # Validate data
                    validate_data(df, detected_columns)

                    # Allow user to select columns with detected defaults
                    st.subheader("Column Selection")
                    pressure_col = st.selectbox(
                        "Select Pressure Column",
                        options=df.columns,
                        index=df.columns.get_loc(detected_columns.get('pressure', df.columns[0]))
                    )
                    revolution_col = st.selectbox(
                        "Select Revolution Column",
                        options=df.columns,
                        index=df.columns.get_loc(detected_columns.get('revolution', df.columns[0]))
                    )

                    # Process the data
                    df[revolution_col] = safe_numeric_conversion(df[revolution_col], revolution_col)
                    df[pressure_col] = safe_numeric_conversion(df[pressure_col], pressure_col)
                    df = df.dropna(subset=[revolution_col, pressure_col])

                    # Calculate torque
                    df['Calculated torque [kNm]'] = df.apply(
                        lambda row: calculate_torque_wrapper(row, pressure_col, revolution_col, machine_params),
                        axis=1
                    )

                    # Create visualization
                    fig = create_torque_visualization(df, revolution_col, pressure_col, machine_params, P_max, nu, anomaly_threshold)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display statistics and explanation
                    display_statistical_summary(df, revolution_col, pressure_col)
                    display_explanation(anomaly_threshold)

                    # Download buttons
                    st.sidebar.markdown("## Download Results")
                    stats_df = create_statistics_df(df, revolution_col, pressure_col)
                    st.sidebar.markdown(
                        get_table_download_link(stats_df, "statistical_analysis.csv", "Download Statistical Analysis"),
                        unsafe_allow_html=True
                    )

        except DataProcessingError as e:
            handle_error(e, "Data Processing Error in Original Page")
        except Exception as e:
            handle_error(e, "Error in Original Analysis Page")

def advanced_page():
    st.title("Advanced Analysis")
    
    data_loader = DataLoader()

    # File uploaders
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications", type=["xlsx", "csv"])

    if machine_specs_file is not None:
        try:
            file_type = machine_specs_file.name.split(".")[-1].lower()
            if file_type == 'xlsx':
                machine_specs = pd.read_excel(machine_specs_file)
            else:
                machine_specs = pd.read_csv(machine_specs_file)
            
            if "Projekt" not in machine_specs.columns:
                st.error("The machine specifications file must contain a 'Projekt' column.")
                st.stop()

            machine_types = machine_specs["Projekt"].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)

            # Get and display machine parameters
            machine_params = get_machine_params(machine_specs, selected_machine)
            if not machine_params:
                st.error("Machine parameters could not be retrieved.")
                st.stop()

            # Display parameters
            params_df = pd.DataFrame([machine_params])
            st.table(params_df)

            # Analysis parameters in sidebar
            st.sidebar.header("Parameter Settings")
            P_max = st.sidebar.number_input("Maximum power (kW)", value=132.0, min_value=1.0, max_value=500.0)
            nu = st.sidebar.number_input("Efficiency coefficient", value=0.7, min_value=0.1, max_value=1.0)
            anomaly_threshold = st.sidebar.number_input("Anomaly threshold (bar)", value=250, min_value=100, max_value=500)
            num_cutting_rings = st.sidebar.number_input("Number of Cutting Rings", value=1, min_value=1, max_value=100)

            if raw_data_file is not None:
                # Load and process data
                file_type = raw_data_file.name.split(".")[-1].lower()
                df, detected_columns = data_loader.load_data(raw_data_file, file_type)

                if df is not None:
                    # Validate data
                    validate_data(df, detected_columns)

                    # Enhanced column selection with detected defaults
                    st.subheader("Select Sensor Columns")
                    
                    columns = {}
                    for sensor_type in ['time', 'pressure', 'revolution', 'advance_rate', 'thrust_force', 'distance']:
                        default_col = detected_columns.get(sensor_type, df.columns[0])
                        try:
                            default_index = df.columns.get_loc(default_col)
                        except KeyError:
                            default_index = 0
                        columns[sensor_type] = st.selectbox(
                            f"Select {sensor_type.replace('_', ' ').title()} Column",
                            options=df.columns,
                            index=default_index
                        )

                    # Convert numeric columns
                    for col in columns.values():
                        df[col] = safe_numeric_conversion(df[col], col)
                    
                    # Process data
                    df = process_advanced_data(df, columns, num_cutting_rings, machine_params)

                    # Create visualizations
                    tabs = st.tabs(["Torque Analysis", "Time Series", "Distance Analysis"])
                    
                    with tabs[0]:
                        fig_torque = create_torque_visualization(df, columns['revolution'], columns['pressure'], 
                                                              machine_params, P_max, nu, anomaly_threshold)
                        st.plotly_chart(fig_torque, use_container_width=True)

                    with tabs[1]:
                        fig_time = create_time_series_visualization(df, columns)
                        st.plotly_chart(fig_time, use_container_width=True)

                    with tabs[2]:
                        fig_distance = create_distance_visualization(df, columns)
                        st.plotly_chart(fig_distance, use_container_width=True)

                    # Display statistics
                    display_advanced_statistical_summary(df, columns)

                    # Download options
                    st.sidebar.markdown("## Download Results")
                    stats_df = create_advanced_statistics_df(df, columns)
                    st.sidebar.markdown(
                        get_table_download_link(stats_df, "advanced_analysis.csv", "Download Analysis Results"),
                        unsafe_allow_html=True
                    )

        except DataProcessingError as e:
            handle_error(e, "Data Processing Error in Advanced Page")
        except Exception as e:
            handle_error(e, "Error in Advanced Analysis Page")

# ----------------------------
# Main Function
# ----------------------------
def main():
    # Page configuration
    st.set_page_config(
        page_title="Herrenknecht Torque Analysis",
        page_icon="https://raw.githubusercontent.com/kilickursat/torque_plot-updated/main/Herrenknecht_logo.svg-1024x695.png",
        layout="wide"
    )

    # Set background color
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

    # Add logo to sidebar
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page selection
    page = st.sidebar.selectbox("Select Page", ("Original Analysis", "Advanced Analysis"))

    if page == "Original Analysis":
        original_page()
    elif page == "Advanced Analysis":
        advanced_page()

if __name__ == "__main__":
    main()
