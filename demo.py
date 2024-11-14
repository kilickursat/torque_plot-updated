# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
import io
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Machine Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
def load_data(file, file_type):
    """
    Loads raw data from a CSV or Excel file.
    """
    try:
        if file_type == "csv":
            df = pd.read_csv(file)
        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type.")
            return None
        logger.info("Raw data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        logger.error(f"Error loading raw data: {e}")
        return None
def load_machine_specs(file, file_type):
    """
    Loads machine specifications from a CSV or Excel file.
    """
    try:
        if file_type == "csv":
            specs_df = pd.read_csv(file)
        elif file_type in ["xlsx", "xls"]:
            specs_df = pd.read_excel(file)
        else:
            st.error("Unsupported file type for machine specifications.")
            return None
        logger.info("Machine specifications loaded successfully.")
        return specs_df
    except Exception as e:
        st.error(f"Error loading machine specifications: {e}")
        logger.error(f"Error loading machine specifications: {e}")
        return None
def find_sensor_columns(df):
    """
    Attempts to find sensor columns in the raw data based on common names.
    Returns a dictionary with sensor types as keys and column names as values.
    """
    sensor_column_map = {
        "time": [
            "Time",
            "Timestamp",
            "DateTime",
            "Date",
            "Zeit",
            "Relativzeit",
            "Uhrzeit",
            "Datum",
            "ts(utc)",
            "Elapsed Time",
            "Elapsed_Time",
            "ElapsedTime",
            "Time_Elapsed"
        ],
        "pressure": [
            "Working pressure [bar]",
            "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26",
            "Pression [bar]",
            "Presión [bar]",
            "Pressure",
            "Pressure [bar]",
            "Working Pressure",
            "cutting wheel.MPU1WPr",
            "MPU1WPr"
        ],
        "revolution": [
            "Revolution [rpm]",
            "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30",
            "Vitesse [rpm]",
            "Revoluciones [rpm]",
            "RPM",
            "Speed",
            "Rotation Speed",
            "cutting wheel.CWSpeed",
            "CWSpeed",
            "cutting wheel"
        ],
        "advance_rate": [
            "Advance Rate",
            "Vorschubgeschwindigkeit",
            "Avance",
            "Rate of Penetration",
            "ROP",
            "Advance [m/min]",
            "Advance [mm/min]",
            "VTgeschw_Z",
            "VTgeschw"
        ],
        "thrust_force": [
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
            "Distance",
            "Chainage",
            "Position",
            "Kette",
            "Station",
            "V34_TL_SR_m_Z",
            "TL_SR_m_Z",
            "SR_m_Z",
            "Weg",
            "weg",
            "Distance_Chainage",
            "Chainage_Position"
        ]
    }

    found_columns = {}
    for sensor, possible_names in sensor_column_map.items():
        found = False
        for name in possible_names:
            if name in df.columns:
                found_columns[sensor] = name
                found = True
                break
        if not found:
            found_columns[sensor] = None
    logger.info("Sensor columns mapping completed.")
    return found_columns
def get_table_download_link(df, filename, link_text):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in CSV format.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href
def calculate_whisker_and_outliers_advanced(series):
    """
    Calculates the lower and upper whiskers (10th and 90th percentiles)
    and identifies outliers based on these whiskers.
    Returns the whiskers and the list of outliers.
    """
    lower_whisker = np.percentile(series, 10)
    upper_whisker = np.percentile(series, 90)
    outliers = series[(series < lower_whisker) | (series > upper_whisker)].unique()
    logger.info("Whiskers and outliers calculated.")
    return lower_whisker, upper_whisker, outliers
def display_statistics(df, revolution_col, pressure_col, thrust_force_col):
    """
    Displays statistical summaries for RPM, Pressure, and Thrust Force.
    """
    st.subheader("Statistical Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Revolution [1/min] Statistics:**")
        st.write(df[revolution_col].describe())
    
    with col2:
        st.write("**Pressure [bar] Statistics:**")
        st.write(df[pressure_col].describe())
    
    with col3:
        st.write("**Thrust Force [kN] Statistics:**")
        st.write(df[thrust_force_col].describe())
    
    logger.info("Statistical summaries displayed.")
def is_valid_machine(params):
    """
    Validates that machine parameters are within acceptable ranges.
    """
    critical_params = ["n1", "n2", "M_cont_value", "M_max_Vg1", "torque_constant"]
    for param in critical_params:
        if params.get(param, 0) == 0 or params.get(param, None) is None:
            return False
    return True
def get_machine_params(specs_df, machine_type):
    """
    Extracts and validates machine parameters for the selected machine type.
    """
    # Filter the DataFrame for the selected machine type
    machine_rows = specs_df[specs_df['Projekt'] == machine_type]
    if machine_rows.empty:
        st.error(f"Machine type '{machine_type}' not found in the specifications file.")
        logger.error(f"Machine type '{machine_type}' not found in the specifications file.")
        return None

    # Extract the first matching row
    machine_data = machine_rows.iloc[0]

    # Define possible column names for each parameter
    machine_column_map = {
        "n1": ["n1[1/min]", "n1 (1/min)", "n1", "RPM Max"],
        "n2": ["n2[1/min]", "n2 (1/min)", "n2", "RPM Min"],
        "M_cont_value": ["M(dauer)[kNm]", "M(dauer)", "Continuous Torque"],
        "M_max_Vg1": ["M(max)[kNm]", "M(max)", "Max Torque"],
        "torque_constant": ["Drehmomentumrechnung [kNm/bar]", "Torque Constant", "Drehmomentumrechnung"]
    }

    # Function to find the correct column name using exact and partial matching
    def find_column(possible_names):
        for name in possible_names:
            # Exact match (case-insensitive)
            exact_matches = specs_df.columns[specs_df.columns.str.lower() == name.lower()]
            if not exact_matches.empty:
                return exact_matches[0]
            # Partial match (contains)
            partial_matches = specs_df.columns[specs_df.columns.str.contains(name, case=False, regex=False)]
            if not partial_matches.empty:
                return partial_matches[0]
        return None

    # Attempt to find each parameter
    found_columns = {}
    for param, possible_names in machine_column_map.items():
        col = find_column(possible_names)
        if col:
            value = machine_data[col]
            # Handle data type conversion
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = None
            found_columns[param] = value
        else:
            found_columns[param] = None  # Explicitly set to None if not found

    # Collect missing parameters
    missing_params = [param for param, value in found_columns.items() if value is None]

    # If any parameters are missing or zero, prompt user to input them
    if missing_params:
        st.warning(
            f"The selected machine '{machine_type}' is missing the following parameters: {', '.join(missing_params)}. "
            f"Please input the missing values below."
        )
        for param in missing_params:
            user_input = st.number_input(
                f"Input value for '{param}':",
                min_value=0.0,
                format="%.2f",
                key=f"{machine_type}_{param}"
            )
            found_columns[param] = user_input

    # Validate parameter values
    if not is_valid_machine(found_columns):
        st.error(
            f"Selected machine '{machine_type}' has invalid or zero parameters even after user input. Please verify the values."
        )
        logger.error(
            f"Machine '{machine_type}' has invalid or zero parameters after user input."
        )
        return None

    # Log the retrieved parameters for debugging
    st.write("### Retrieved Machine Parameters")
    params_df = pd.DataFrame([found_columns])
    st.write(params_df)
    logger.info(f"Machine parameters for '{machine_type}' retrieved successfully.")

    return found_columns
def advanced_page():
    st.title("Advanced Machine Data Analysis")

    # File uploaders for batch data
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader(
        "Upload Machine Specifications (MM-Baureihenliste): CSV or XLSX",
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
                st.error("Machine parameters could not be retrieved. Please ensure all required parameters are present.")
                st.stop()

        except Exception as e:
            st.error(f"An error occurred while processing the machine specifications: {e}")
            logger.error(f"Error processing machine specifications: {e}")
            st.stop()
    else:
        st.warning("Please upload the Machine Specifications file to proceed.")
        st.stop()

    # Sidebar for user inputs
    st.sidebar.header("Parameter Settings")
    P_max = st.sidebar.number_input(
        "Maximum Power (kW)", value=132.0, min_value=1.0, max_value=500.0
    )
    nu = st.sidebar.number_input(
        "Efficiency Coefficient", value=0.7, min_value=0.1, max_value=1.0
    )
    anomaly_threshold = st.sidebar.number_input(
        "Anomaly Threshold (bar)", value=250, min_value=100, max_value=500
    )
    num_cutting_rings = st.sidebar.number_input(
        "Number of Cutting Rings", value=1, min_value=1, max_value=100
    )

    if raw_data_file is not None:
        # Load raw data
        file_type = raw_data_file.name.split(".")[-1].lower()
        df = load_data(raw_data_file, file_type)

        if df is not None:
            # Find sensor columns
            sensor_columns = find_sensor_columns(df)

            # Allow user to select columns if not found or adjust selections
            st.subheader("Select Sensor Columns")

            # Add a selectbox to let the user specify if the time column is numeric or datetime
            time_column_type_user = st.selectbox(
                "Select Time Column Type",
                options=["Numeric", "Datetime"],
                index=0,
                help="Choose how to interpret the Time column based on your raw data. 'Numeric' treats it as elapsed time units, while 'Datetime' parses it as dates and times."
            )

            # Function to map sensor columns with manual override
            def map_sensor_column(sensor_type, sensor_column_map):
                possible_names = sensor_column_map.get(sensor_type, [])
                for name in possible_names:
                    if name in df.columns:
                        return name
                # Attempt partial matching
                for name in possible_names:
                    matching_cols = [col for col in df.columns if name.lower() in col.lower()]
                    if matching_cols:
                        return matching_cols[0]
                # If not found, prompt user to select manually
                st.warning(f"Sensor column for '{sensor_type}' not found automatically. Please select it manually.")
                return st.selectbox(f"Select {sensor_type.capitalize()} Column", options=df.columns)

            # Define sensor column mappings
            sensor_column_map = {
                "time": [
                    "Time",
                    "Timestamp",
                    "DateTime",
                    "Date",
                    "Zeit",
                    "Relativzeit",
                    "Uhrzeit",
                    "Datum",
                    "ts(utc)",
                    "Elapsed Time",
                    "Elapsed_Time",
                    "ElapsedTime",
                    "Time_Elapsed"
                ],
                "pressure": [
                    "Working pressure [bar]",
                    "AzV.V13_SR_ArbDr_Z | DB 60.DBD 26",
                    "Pression [bar]",
                    "Presión [bar]",
                    "Pressure",
                    "Pressure [bar]",
                    "Working Pressure",
                    "cutting wheel.MPU1WPr",
                    "MPU1WPr"
                ],
                "revolution": [
                    "Revolution [rpm]",
                    "AzV.V13_SR_Drehz_nach_Abgl_Z | DB 60.DBD 30",
                    "Vitesse [rpm]",
                    "Revoluciones [rpm]",
                    "RPM",
                    "Speed",
                    "Rotation Speed",
                    "cutting wheel.CWSpeed",
                    "CWSpeed",
                    "cutting wheel"
                ],
                "advance_rate": [
                    "Advance Rate",
                    "Vorschubgeschwindigkeit",
                    "Avance",
                    "Rate of Penetration",
                    "ROP",
                    "Advance [m/min]",
                    "Advance [mm/min]",
                    "VTgeschw_Z",
                    "VTgeschw"
                ],
                "thrust_force": [
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
                    "Distance",
                    "Chainage",
                    "Position",
                    "Kette",
                    "Station",
                    "V34_TL_SR_m_Z",
                    "TL_SR_m_Z",
                    "SR_m_Z",
                    "Weg",
                    "weg",
                    "Distance_Chainage",
                    "Chainage_Position"
                ]
            }

            # Map each sensor
            time_col = map_sensor_column("time", sensor_column_map)
            pressure_col = map_sensor_column("pressure", sensor_column_map)
            revolution_col = map_sensor_column("revolution", sensor_column_map)
            advance_rate_col = map_sensor_column("advance_rate", sensor_column_map)
            thrust_force_col = map_sensor_column("thrust_force", sensor_column_map)
            distance_col = map_sensor_column("distance", sensor_column_map)

            # Display selected columns for confirmation
            st.write("### Selected Sensor Columns")
            st.write(f"**Time Column:** {time_col}")
            st.write(f"**Pressure Column:** {pressure_col}")
            st.write(f"**Revolution Column:** {revolution_col}")
            st.write(f"**Advance Rate Column:** {advance_rate_col}")
            st.write(f"**Thrust Force Column:** {thrust_force_col}")
            st.write(f"**Distance/Chainage Column:** {distance_col}")

            # Validate required columns
            required_columns = [time_col, pressure_col, revolution_col, advance_rate_col, thrust_force_col, distance_col]
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                st.error(f"The following required columns are missing from the raw data: {', '.join(missing_required)}")
                logger.error(f"Missing required columns: {', '.join(missing_required)}")
                st.stop()

            # Ensure distance column is appropriately parsed
            df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
            if df[distance_col].isnull().all():
                st.error(
                    f"The selected distance/chainage column '{distance_col}' cannot be converted to numeric values."
                )
                logger.error(f"Distance/Chainage column '{distance_col}' conversion failed.")
                st.stop()

            # Handle missing values
            missing_distance = df[distance_col].isnull().sum()
            if missing_distance > 0:
                st.warning(f"There are {missing_distance} missing values in the distance/chainage column. These rows will be dropped.")
                df = df.dropna(subset=[distance_col])
                logger.warning(f"Dropped {missing_distance} rows due to missing distance/chainage values.")

            # Display the maximum value in the distance/chainage column for debugging
            max_distance_value = df[distance_col].max()
            st.write(f"**Maximum value in the distance/chainage column (`{distance_col}`):** {max_distance_value}")

            # Assign and process Time_unit
            df = process_time_column(df, time_col, time_column_type_user)

            # Debugging: Check DataFrame after Time_unit processing
            st.write("### Data After Time_unit Processing")
            st.write(df.head())
            st.write("### Time_unit Column Exists:", "Time_unit" in df.columns)
            if "Time_unit" in df.columns:
                st.write("### Time_unit Sample:", df["Time_unit"].head())

            # Ensure numeric columns are numeric and handle missing values
            numeric_columns = [pressure_col, revolution_col, advance_rate_col, thrust_force_col]
            df = validate_and_convert_columns(df, numeric_columns)
            st.write("### Data After Numeric Conversion and Dropping NaNs")
            st.write(df.head())
            st.write(f"**DataFrame Shape:** {df.shape}")

            if df.empty:
                st.error("No data available after converting numeric columns and dropping missing values.")
                logger.error("DataFrame is empty after numeric conversion and dropping NaNs.")
                st.stop()

            # Remove rows where revolution is zero to avoid division by zero
            df = df[df[revolution_col] != 0]
            st.write(f"### Data After Removing Rows with {revolution_col} = 0")
            st.write(f"**DataFrame Shape:** {df.shape}")
            st.write(df.head())

            if df.empty:
                st.error(f"No data remaining after removing rows where {revolution_col} is zero.")
                logger.error(f"DataFrame is empty after removing rows with {revolution_col} = 0.")
                st.stop()

            # Calculate Metrics
            df = calculate_metrics(df, machine_params, {
                'pressure': pressure_col,
                'revolution': revolution_col,
                'advance_rate': advance_rate_col,
                'thrust_force': thrust_force_col
            }, num_cutting_rings)

            st.write("### Data After Calculating Metrics")
            st.write(df.head())
            st.write(f"**DataFrame Shape:** {df.shape}")

            if df.empty:
                st.error("No data available after calculating metrics.")
                logger.error("DataFrame is empty after calculating metrics.")
                st.stop()

            # RPM Statistics
            rpm_stats = df[revolution_col].describe()
            rpm_max_value = rpm_stats["max"]
            st.sidebar.write(
                f"Recommended value for x-axis based on the Max RPM in Data: {rpm_max_value:.2f}"
            )

            # Allow user to set x_axis_max with validation
            x_axis_max = st.sidebar.number_input(
                "X-axis Maximum",
                value=float(rpm_max_value),
                min_value=1.0,
                max_value=float(rpm_max_value * 1.2),
                help="Set the maximum value for the X-axis (Revolution)."
            )

            # Filter data points between n2 and n1 rpm
            n2 = machine_params.get("n2", df[revolution_col].min())
            n1 = machine_params.get("n1", df[revolution_col].max())
            df = df[
                (df[revolution_col] >= n2)
                & (df[revolution_col] <= n1)
            ]

            st.write(f"### Data After Filtering RPM between {n2} and {n1}")
            st.write(f"**DataFrame Shape:** {df.shape}")
            st.write(df.head())

            if df.empty:
                st.error("No data available after filtering RPM range.")
                logger.error("DataFrame is empty after filtering RPM range.")
                st.stop()

            # Recalculate whiskers and outliers
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

            # Function to calculate M_max_Vg2
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

            # Create Subplots for Torque Analysis
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
            if not normal_data.empty:
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

            if not anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data[revolution_col],
                        y=anomaly_data["Calculated torque [kNm]"],
                        mode="markers",
                        name=f"Anomaly (Pressure ≥ {anomaly_threshold} bar)",
                        marker=dict(color="red", symbol="x", size=10),
                    )
                )

            if not torque_outlier_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=torque_outlier_data[revolution_col],
                        y=torque_outlier_data["Calculated torque [kNm]"],
                        mode="markers",
                        name="Torque Outliers",
                        marker=dict(color="orange", symbol="diamond", size=10),
                    )
                )

            if not rpm_outlier_data.empty:
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

            # Define the additional statistical metrics
            additional_stats = {
                "Advance Rate": df[advance_rate_col].describe(),
                "Penetration Rate (Calculated)": df["Calculated Penetration Rate"].describe(),
                "Thrust Force at the Cutting Head": df[thrust_force_col].describe(),
                "Thrust Force per Cutting Ring": df["Thrust Force per Cutting Ring"].describe()
            }

            for feature_name, stats in additional_stats.items():
                st.write(f"**{feature_name} Statistics:**")
                st.write(stats)

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

            # Calculate rolling means for each feature
            for feature in features:
                df[f"{feature['column']}_mean"] = df[feature['column']].rolling(window=window_size, min_periods=1).mean()

            # Create subplots with 2 rows per feature: one for original data, one for mean
            fig_time = make_subplots(
                rows=2*num_features,  # Two rows per feature
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,  # Reduced spacing for a cleaner look
                subplot_titles=None  # No subplot titles
            )

            # Iterate through each feature and add traces
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

                # Rolling Mean Plot on even rows
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
                    # Update y-axis for rolling mean
                    fig_time.update_yaxes(title_text=f"{feature['display_name']} - Rolling Mean", row=2*i, col=1)

            # Update overall layout
            fig_time.update_layout(
                xaxis_title=f"Time",
                height=300 * 2 * num_features,  # 300 pixels per subplot row
                showlegend=False,
                title_text="Features over Time (Original and Rolling Mean)",  # Main plot title
            )

            # Display the plot
            st.plotly_chart(fig_time, use_container_width=True)

            # --------------------- Features over Distance/Chainage Visualization ---------------------

            st.subheader("Features over Distance/Chainage")

            # Define the features with their display names and colors
            features_distance = [
                {"column": advance_rate_col, "display_name": "Advance Rate", "color": "blue"},
                {"column": "Calculated Penetration Rate", "display_name": "Penetration Rate", "color": "green"},
                {"column": thrust_force_col, "display_name": "Thrust Force", "color": "red"},
                {"column": "Thrust Force per Cutting Ring", "display_name": "Thrust Force per Cutting Ring", "color": "orange"},
                {"column": revolution_col, "display_name": "Revolution", "color": "purple"},
                {"column": pressure_col, "display_name": "Working Pressure", "color": "cyan"},
                {"column": "Calculated torque [kNm]", "display_name": "Calculated Torque [kNm]", "color": "magenta"},
            ]

            # Define the number of features
            num_features_distance = len(features_distance)

            # Rolling Window Slider for Distance
            window_size_distance = st.sidebar.slider(
                "Select Rolling Window Size for Mean Calculation (Distance)",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Adjust the window size to smooth the data. A larger window provides a smoother mean."
            )

            # Toggle for Mean Lines
            show_means_distance = st.checkbox(
                "Show Rolling Mean Values (Distance)",
                value=True,
                help="Toggle the visibility of rolling mean lines for distance-based features."
            )

            # Sort the dataframe by Distance/Chainage to avoid zigzag lines
            df_distance_sorted = df.sort_values(by=distance_col)

            # Calculate rolling means for each feature
            for feature in features_distance:
                if feature['column'] in df_distance_sorted.columns:
                    df_distance_sorted[f"{feature['column']}_distance_mean"] = df_distance_sorted[feature['column']].rolling(
                        window=window_size_distance, min_periods=1
                    ).mean()
                else:
                    st.warning(f"Column '{feature['column']}' not found in the dataset.")
                    df_distance_sorted[f"{feature['column']}_distance_mean"] = np.nan

            # Create subplots with 2 rows per feature: one for original data, one for mean
            fig_distance = make_subplots(
                rows=2*num_features_distance,  # Two rows per feature
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,  # Reduced spacing for a cleaner look
                subplot_titles=None  # No subplot titles
            )

            # Iterate through each feature and add traces
            for i, feature in enumerate(features_distance, start=1):
                # Original Feature Plot on odd rows
                if feature['column'] in df_distance_sorted.columns:
                    fig_distance.add_trace(
                        go.Scatter(
                            x=df_distance_sorted[distance_col],
                            y=df_distance_sorted[feature["column"]],
                            mode="lines",
                            name=feature["display_name"],
                            line=dict(color=feature["color"]),
                        ),
                        row=2*i-1,
                        col=1,
                    )
                    # Update y-axis for original feature
                    fig_distance.update_yaxes(title_text=feature["display_name"], row=2*i-1, col=1)

                    # Rolling Mean Plot on even rows
                    if show_means_distance:
                        fig_distance.add_trace(
                            go.Scatter(
                                x=df_distance_sorted[distance_col],
                                y=df_distance_sorted[f"{feature['column']}_distance_mean"],
                                mode="lines",
                                name=f"{feature['display_name']} Rolling Mean",
                                line=dict(color=feature["color"], dash="dash"),
                            ),
                            row=2*i,
                            col=1,
                        )
                        # Update y-axis for rolling mean
                        fig_distance.update_yaxes(title_text=f"{feature['display_name']} Rolling Mean", row=2*i, col=1)
                else:
                    st.warning(f"Column '{feature['column']}' not found in the dataset.")

            # Update overall layout
            fig_distance.update_layout(
                xaxis_title=f"Distance/Chainage",
                height=300 * 2 * num_features_distance,  # 300 pixels per subplot row
                showlegend=False,
                title_text="Features over Distance/Chainage (Original and Rolling Mean)",  # Main plot title
            )

            # Display the plot
            st.plotly_chart(fig_distance, use_container_width=True)

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
                - **Calculated Torque [kNm]**: Represents the torque calculated based on machine parameters and operational data.

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

    # Helper functions used within advanced_page

    def process_time_column(df, time_col, time_type):
        """
        Processes the time column based on the user-selected type (Numeric or Datetime).
        Assigns a 'Time_unit' column for consistent plotting.
        """
        if time_type == "Numeric":
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            if df[time_col].isnull().all():
                st.error(f"The selected time column '{time_col}' cannot be converted to numeric values.")
                logger.error(f"Time column '{time_col}' conversion to numeric failed.")
                st.stop()
            df["Time_unit"] = df[time_col]
            # Sort the dataframe by Time_unit
            df = df.sort_values("Time_unit")
            # Calculate min and max time
            min_time_unit = df["Time_unit"].min()
            max_time_unit = df["Time_unit"].max()
            # Display the time range in numeric format
            st.write(f"**Data Time Range:** {min_time_unit:.2f} to {max_time_unit:.2f} units")
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
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if df[time_col].isnull().all():
                    st.error(f"The selected time column '{time_col}' cannot be converted to datetime values.")
                    logger.error(f"Time column '{time_col}' conversion to datetime failed.")
                    st.stop()
                df["Time_unit"] = df[time_col]
                # Sort the dataframe by Time_unit
                df = df.sort_values("Time_unit")
                # Calculate min and max time
                min_time_unit = df["Time_unit"].min().to_pydatetime()
                max_time_unit = df["Time_unit"].max().to_pydatetime()
                # Display the time range
                st.write(f"**Data Time Range:** {min_time_unit} to {max_time_unit}")
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
            except Exception as e:
                st.error(f"Error processing datetime for time column '{time_col}': {e}")
                logger.error(f"Error processing datetime for time column '{time_col}': {e}")
                st.stop()
        return df
if __name__ == "__main__":
    advanced_page()
