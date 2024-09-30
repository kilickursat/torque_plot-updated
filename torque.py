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

def display_column_selection(df):
    st.write("Please select the columns for pressure and revolution data:")
    
    found_columns = find_sensor_columns(df)
    
    pressure_col = st.selectbox(
        "Select pressure column",
        options=[""] + list(df.columns),
        index=0 if "pressure" not in found_columns else list(df.columns).index(found_columns["pressure"]) + 1,
        key="pressure_column"
    )
    
    revolution_col = st.selectbox(
        "Select revolution column",
        options=[""] + list(df.columns),
        index=0 if "revolution" not in found_columns else list(df.columns).index(found_columns["revolution"]) + 1,
        key="revolution_column"
    )
    
    return pressure_col, revolution_col

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

    # Add logo to the sidebar
    add_logo()
    
    # Add this line to create a div for sidebar content
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.title("TorqueVision: Herrenknecht's Advanced Analysis App")
    st.sidebar.markdown("**Created by Kursat Kilic - Geotechnical Digitalization**")
    
    # Add this line at the end of your main function
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # File uploaders for batch data
    raw_data_file = st.file_uploader("Upload Raw Data (CSV or XLSX)", type=["csv", "xlsx"])
    machine_specs_file = st.file_uploader("Upload Machine Specifications XLSX", type="xlsx")

    # Load machine specs if available
    if machine_specs_file is not None:
        try:
            machine_specs = load_machine_specs(machine_specs_file)
            machine_types = machine_specs['Projekt'].unique()
            selected_machine = st.sidebar.selectbox("Select Machine Type", machine_types)
            
            machine_params = get_machine_params(machine_specs, selected_machine)
            
            # Display loaded parameters in a table
            st.write("**Loaded Machine Parameters:**")
            params_df = pd.DataFrame([machine_params])
            
            st.dataframe(params_df)
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
    df = load_data_file(raw_data_file)
    
    if df is not None:
        # Display columns with hover-over functionality
        display_columns_with_hover(df)

        # Allow user to select columns
        pressure_col, revolution_col = display_column_selection(df)
        
# ... (previous code for column selection)

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
    x_axis_max = st.sidebar.number_input("X-axis maximum", value=rpm_max_value, min_value=1.0, max_value=100.0)

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

    # Add vertical lines at elbow points
    ax.plot([elbow_rpm_max, elbow_rpm_max], [0, machine_params['M_max_Vg1']], color='purple', linestyle=':', linewidth=3)
    ax.plot([elbow_rpm_cont, elbow_rpm_cont], [0, machine_params['M_cont_value']], color='orange', linestyle=':', linewidth=3)

    # Add vertical line for n1
    ax.plot([machine_params['n1'], machine_params['n1']], [0, machine_params['M_cont_value']], color='black', linestyle='--', linewidth=2)

    # Plot calculated torque vs RPM, differentiating between normal, anomaly, and outlier points
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

    # Add horizontal lines for the torque whiskers
    ax.axhline(y=torque_upper_whisker, color='gray', linestyle='--', linewidth=1, label='Torque Upper Whisker')
    ax.axhline(y=torque_lower_whisker, color='gray', linestyle=':', linewidth=1, label='Torque Lower Whisker')

    # Set plot limits and labels
    ax.set_xlim(0, x_axis_max)
    ax.set_ylim(0, max(60, df['Calculated torque [kNm]'].max() * 1.1))
    ax.set_xlabel('Revolution [1/min]')
    ax.set_ylabel('Torque [kNm]')
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
                else:
                    st.warning("Please select both pressure and revolution columns to proceed with the analysis.")
            else:
                st.error("Failed to load the file. Please check the format and try again.")
        else:
            st.info("Please upload a Raw Data file to begin the analysis.")
    else:
        st.warning("Please upload Machine Specifications XLSX file.")

if __name__ == "__main__":
    main()
