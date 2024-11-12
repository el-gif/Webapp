import io
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # to disable Ctrl+C crashing python when having scipy.interpolate imported (disables Fortrun runtime library from intercepting Ctrl+C signal and lets it to the Python interpreter)
from openpyxl import Workbook
from shiny import ui, App, reactive
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Marker, MarkerCluster, WidgetControl, FullScreenControl, AwesomeIcon, Heatmap
from ipywidgets import SelectionSlider, Play, VBox, jslink, Layout, HTML  # pip install ipywidgets==7.6.5, because version 8 has an issue with popups (https://stackoverflow.com/questions/75434737/shiny-for-python-using-add-layer-for-popus-from-ipyleaflet)
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d  # pip install scipy==1.13.1, interp2d much faster than RegularGridInterpolator, even if deprecated
import base64
from ecmwf.opendata import Client
from branca.colormap import linear
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
starttime = time.time()

# Define initial variables
initial = 0
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# Path to the stored weather data
wind_file = "data/weather_forecast/data_europe.grib2"
# Set time interval for updates (in seconds)
time_threshold = 24 * 3600  # 24 hours in seconds
overwrite = 0  # force overwriting of data

# Initialise ECMWF client
client = Client(
    source="ecmwf",
    model="ifs",
    resol="0p25"
)

if os.getenv("RENDER") or os.getenv("WEBSITE_HOSTNAME"):  # for Render or Azure Server
    # for saving storage on server, only 4 time steps
    step_selection = [0, 3, 6, 9]
else:
    # all available steps up to 7 days (168 hours)
    step_selection = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
                      63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120,
                      123, 126, 129, 132, 135, 138, 141, 144, 150, 156, 162, 168]

# all available steps
# step_selection = [
#     0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
#     63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120,
#     123, 126, 129, 132, 135, 138, 141, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204,
#     210, 216, 222, 228, 234, 240
# ]

# Function to check the age of the file, to avoid a cron-job, which is a paid feature on most Cloud Computing platforms
def is_data_stale(wind_file, time_threshold):
    if not os.path.exists(wind_file):
        # File does not exist, data must be fetched
        return True
    # Determine the age of the file (in seconds since the last modification)
    file_age = time.time() - os.path.getmtime(wind_file)
    # Check if the file is older than the specified threshold
    return file_age > time_threshold

if is_data_stale(wind_file, time_threshold) or overwrite:
    print("Data is outdated, does not exist, or should be overwritten. Fetching new data.")
    # Fetching API data since they are either missing or older than 10 hours
    result = client.retrieve(  # retrieve data worldwide, because no area tag available (https://github.com/ecmwf/ecmwf-opendata, https://www.ecmwf.int/en/forecasts/datasets/open-data)
        type="fc",
        param=["100v", "100u"],  # U- and V-components of wind speed
        target=wind_file,
        time=0,  # Forecast time (model run at 00z)
        step=step_selection
    )
    print("New data has been successfully fetched and saved.")
else:
    print("Data is current and will be used.")

# Load the wind data (Grib2 file)
ds = xr.open_dataset(wind_file, engine='cfgrib')

# Filter the data for Europe and extract relevant columns
ds_filtered = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
lats = ds_filtered['latitude'].values
lons = ds_filtered['longitude'].values
u = ds_filtered['u100'].values
v = ds_filtered['v100'].values
valid_times = ds_filtered['valid_time'].values

# Filter the data for Europe and extract relevant columns
df_filtered = pd.read_parquet("data/WPPs/Global-Wind-Power-Tracker-Europe.parquet") # 0.7 seconds when WPPs already regionally filtered and stored as parquet file. As unfiltered excel file it takes 11 seconds
df_filtered = df_filtered.iloc[::10]
print(len(df_filtered))
df_filtered['ID'] = list(range(2, len(df_filtered) + 2))
ids = df_filtered['ID'].values
lats_plants = df_filtered['Latitude'].values
lons_plants = df_filtered['Longitude'].values
capacities = df_filtered['Capacity (MW)'].values
project_names = df_filtered['Project Name'].values
statuses = df_filtered['Status'].values
operators = df_filtered['Operator'].values
owners = df_filtered['Owner'].values

# Interpolation period and interval
start_time = valid_times[0]
end_time = valid_times[-1]
total_hours = int((end_time - start_time) / np.timedelta64(1, 'h'))
step_size = np.timedelta64(1, 'h')
step_size_hours = step_size / np.timedelta64(1, 'h')
valid_times_interpol = [start_time + i * step_size for i in range(int(total_hours / step_size_hours))]

# Calculate total wind speed and convert to 3D array
total_selection = np.array([np.sqrt(u_value**2 + v_value**2) for u_value, v_value in zip(u, v)])

# Define the power curve (wind speed and output)
wind_speeds = np.arange(0, 25.5, 0.5)
power_output = [0]*7 + [35, 80, 155, 238, 350, 474, 630, 802, 1018, 1234, 1504, 1773, 2076, 2379, 2664, 2948, 3141, 3334, 3425, 3515, 3546, 3577, 3586, 3594, 3598, 3599] + [3600]*18
max_capacity = 3600
power_output_normalised = [value / max_capacity for value in power_output]
power_curve_normalised = interp1d(wind_speeds, power_output_normalised, kind='cubic', fill_value="extrapolate")

# Function for temporal interpolation
def interpolation(list):
    list_interpol = []
    for i in range(len(list) - 1):
        step_start = valid_times[i]
        step_end = valid_times[i + 1]
        list_start = list[i]
        list_end = list[i + 1]
        list_interpol.append(list_start)
        interval = step_end - step_start
        for step in np.arange(step_start + step_size, step_end, step_size):
            factor = (step - step_start) / interval
            interpolated_value = (1 - factor) * list_start + factor * list_end
            list_interpol.append(interpolated_value)
    list_interpol.append(list[-1])
    return list_interpol

# Interpolation
total_selection_interpol = interpolation(total_selection)

valid_times_interpol = []
steps = int(total_hours / step_size_hours)
for i in range(steps):
    valid_times_interpol.append(start_time + i * step_size)

# Shiny app user interface with download button
app_ui = ui.page_fluid(
    output_widget("map"),
    ui.head_content(
        ui.tags.script("""
            Shiny.addCustomMessageHandler("download", function(message) {
                var link = document.createElement('a');
                link.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + message.data;
                link.download = message.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
        """)
    ),
    ui.head_content(ui.tags.link(rel="icon", type="image/png", href="/www/WPP_icon2.png")),
    ui.panel_title(window_title="Wind Power Forecast", title="")
)

# server function
def server(input, output, session):

    # Function for downloading production data (with conversion from datetime64 to datetime)
    async def download_data_for_marker(plant_index):
        
        lat_plant = lats_plants[plant_index]
        lon_plant = lons_plants[plant_index]

        productions = []
        wind_speeds_at_points = []
        for step_index in range(len(valid_times_interpol)):
            spatial_interpolator = interp2d(lons, lats, total_selection_interpol[step_index], kind='cubic')
            wind_speeds_at_points.append(spatial_interpolator(lon_plant, lat_plant)[0])
            productions.append(np.abs(power_curve_normalised(wind_speeds_at_points[-1]) * capacities[plant_index]))  # absolute values because from time to time a -0.01 MW value arises
        
        # Create Workbook
        wb = Workbook()
        ws = wb.active
        # Truncate title to a maximum of 31 characters to avoid warning
        ws.title = f"Prod Data {project_names[plant_index][:25]}"

        # Headers
        headers = ["Time Step", "Wind speed (m/s)", "Production (MW)"]
        ws.append(headers)

        # Add production data per time step
        for time, wind_speed, production in zip(valid_times_interpol, wind_speeds_at_points, productions):
            time_as_datetime = pd.to_datetime(time).to_pydatetime()
            ws.append([time_as_datetime, wind_speed, production])  # Ensure wind speed and production values are correctly added

        # Save in buffer and return
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()

    @output
    @render_widget
    def map():
        m = Map(
            center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            zoom=5,
            layout=Layout(width='100%', height='100vh'),
            scroll_wheel_zoom=True
        )
        print(time.time()-starttime)
        markers = []
        for index, (lat, lon, name, capacity, status, operator, owner) in enumerate(zip(lats_plants, lons_plants, project_names, capacities, statuses, operators, owners)):

            # if status == "operating":
            #     color = "green"
            # elif status in ["construction", "pre-construction", "announced"]:
            #     color = "orange"
            # else:
            #     color = "red"

            # icon = AwesomeIcon(
            #     name="circle",
            #     marker_color=color
            # )

            popup_content = HTML(
                value=f"<strong>Project Name:</strong> {name}<br>"
                    f"<strong>Status:</strong> {status}<br>"
                    f"<strong>Capacity:</strong> {capacity} MW<br>"
                    f"<strong>Operator:</strong> {operator}<br>"
                    f"<strong>Owner:</strong> {owner}<br>"
                    f"<strong>Wind speed forecast:</strong> select forecast step<br>"
                    f"<strong>Production forecast:</strong> select forecast step<br>"
                    f"<button onclick='Shiny.setInputValue(\"selected_plant_index\", {index})'>Download Forecast</button>"
            )

            marker = Marker(
                location=(lat, lon),
                popup=popup_content,
                #icon=icon, # takes too long
                rise_offset=True,
                draggable=False
            )
            markers.append(marker)
        print(time.time()-starttime)
        # Cluster erstellen und zur Karte hinzufügen
        marker_cluster = MarkerCluster(markers=markers)
        m.add(marker_cluster)


        # Slider for time steps
        play = Play(min=0, max=total_hours, step=step_size_hours, value=0, interval=500, description='Time Step')
        slider = SelectionSlider(options=valid_times_interpol, value=valid_times_interpol[0], description='Time')
        jslink((play, 'value'), (slider, 'index'))
        slider_box = VBox([play, slider])
        m.add(WidgetControl(widget=slider_box, position='topright'))

        # Map update function based on slider
        def update_map(change):
            pass
            time_step = slider.value
            step_index = int((time_step - start_time) / step_size)

            spatial_interpolator = interp2d(lons, lats, total_selection_interpol[step_index], kind='cubic')
            wind_speeds_at_points = [spatial_interpolator(lon, lat)[0] for lon, lat in zip(lons_plants, lats_plants)]

            productions = np.abs(power_curve_normalised(wind_speeds_at_points) * capacities)

            # Update marker pop-ups with production values
            for index, (marker, name, status, capacity, operator, owner, wind_speed, production) in enumerate(zip(marker_cluster.markers, project_names, statuses, capacities, operators, owners, wind_speeds_at_points, productions)):
                marker.popup.value = f"<strong>Project Name:</strong> {name}<br>" \
                                     f"<strong>Status:</strong> {status}<br>" \
                                     f"<strong>Capacity:</strong> {capacity} MW<br>" \
                                     f"<strong>Operator:</strong> {operator}<br>" \
                                     f"<strong>Owner:</strong> {owner}<br>" \
                                     f"<strong>Wind speed forecast:</strong> {wind_speed:.2f} m/s<br>" \
                                     f"<strong>Production forecast:</strong> {production:.2f} MW" \
                                     f"<button onclick='Shiny.setInputValue(\"selected_plant_index\", {index})'>Download Forecast</button>"
                
            # Prepare heatmap data for "operating" wind plants
            heatmap_data = [(lat, lon, prod) for lat, lon, prod, status in zip(lats_plants, lons_plants, productions, statuses) if status == "operating"]

            # Remove existing heatmap layer if any and add new one
            for layer in m.layers:
                if isinstance(layer, Heatmap):
                    m.remove(layer)
            heatmap = Heatmap(locations=heatmap_data, radius=5, blur=5, max_zoom=10)
            m.add(heatmap)
        
        global initial
        # Initialise the map with the first time step
        if initial == 0:
            update_map(None)
            initial = 1
        
        # Observe slider value changes and update map
        slider.observe(update_map, names='value')

        # Add FullScreenControl to map
        m.add(FullScreenControl())

        return m

    # Function to encode in Base64 and send the message
    @reactive.Effect
    @reactive.event(input.selected_plant_index)
    async def trigger_download():
        plant_index = input.selected_plant_index()
        # If a plant index has been selected, download the file
        if plant_index is not None:
            # Retrieve the file as bytes
            file_data = await download_data_for_marker(plant_index)
            # Encode file in Base64
            file_data_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Send message to the client with Base64-encoded JSON file
            await session.send_custom_message("download", {
                "data": file_data_base64,
                "filename": f"production_{project_names[plant_index]}.xlsx"
            })

# Define absolute path to `www` folder
path_www = os.path.join(os.path.dirname(__file__), "www")

# Start the app with the `www` directory as static content
app = App(app_ui, server, static_assets={"/www": path_www})

if __name__ == "__main__":
    # Check if the variable `RENDER` is set to detect if the app is running on Render
    if os.getenv("RENDER") or os.getenv("WEBSITE_HOSTNAME"):  # for Render or Azure Server
        host = "0.0.0.0"  # For Render or other external deployments
    else:
        host = "127.0.0.1"  # For local development (localhost)

    app.run(host=host, port=8000)  # port binding: set the server port to 8000, because this is what Azure expects
