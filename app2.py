import io
import os
from openpyxl import Workbook
from shiny import ui, App, reactive, render
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Marker, MarkerCluster, WidgetControl, FullScreenControl, Heatmap
from ipywidgets import SelectionSlider, Play, VBox, jslink, Layout, HTML
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d, interp2d # pip install scipy==1.13.1, because interp2d was removed in 1.14.0, but performs much better than RegularGridInterpolator

# Define initial variables
initial = 0
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# Load the WPPs (Excel file)
WPP_file = "Global-Wind-Power-Tracker-June-2024.xlsx"
df = pd.read_excel(WPP_file, sheet_name='Data') # nrows=50 for accelerated loading

# Filter the data for Europe and extract relevant columns
df_filtered = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) & (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]
lats_plants = df_filtered['Latitude'].values
lons_plants = df_filtered['Longitude'].values
capacities = df_filtered['Capacity (MW)'].values
project_names = df_filtered['Project Name'].values
start_years = df_filtered['Start year'].values
operators = df_filtered['Operator'].values
owners = df_filtered['Owner'].values

# Load the wind data (Grib2 file)
wind_file = "data_europe.grib2"
ds = xr.open_dataset(wind_file, engine='cfgrib')

# Filter the data for Europe and extract relevant columns
ds_filtered = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
lats = ds_filtered['latitude'].values
lons = ds_filtered['longitude'].values
u = ds_filtered['u100'].values
v = ds_filtered['v100'].values
valid_times = ds_filtered['valid_time'].values

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
    valid_times_interpol.append(start_time + i*step_size)

# Shiny app user interface with download button
app_ui = ui.page_fluid(
    ui.tags.div(
        ui.download_button("download_data", "Download Production at this time step as Excel"),
        style="position: absolute; bottom: 10px; left: 10px; background-color: white; border: none; padding: 5px; z-index: 1000;"
    ),
    ui.tags.div(
        output_widget("map"),
        style="position: relative; top: 10px; left: 10px; padding: 5px; z-index: 0;"
    ),
    ui.head_content(ui.tags.link(rel="icon", type="image/png", href="/www/WPP_icon.png")),
    ui.panel_title(window_title="Wind Power Forecast", title="")
)

# Shiny server logic
def server(input, output, session):

    # Reaktive Variable zur Speicherung der Produktionsdaten
    reactive_data = reactive.Value(None)

    # Render the leaflet widget with render_widget
    @output
    @render_widget
    def map():
        m = Map(
            center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            zoom=5,
            layout=Layout(width='100%', height='100vh'),
            scroll_wheel_zoom=True
        )

        markers = []
        for lat, lon, name, capacity, operator, owner in zip(lats_plants, lons_plants, project_names, capacities, operators, owners):
            popup_content = HTML(
                value=f"<strong>Project Name:</strong> {name}<br>"
                      f"<strong>Capacity:</strong> {capacity} MW<br>"
                      f"<strong>Operator:</strong> {operator}<br>"
                      f"<strong>Owner:</strong> {owner}<br>"
                      f"<strong>Wind Speed:</strong> select forecast step<br>"
                      f"<strong>Production forecast:</strong> select forecast step"
            )
            marker = Marker(location=(lat, lon), popup=popup_content)
            markers.append(marker)
        
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
            time_step = slider.value
            step_index = int((time_step - start_time) / step_size)

            spatial_interpolator = interp2d(lons, lats, total_selection_interpol[step_index], kind='cubic')
            wind_speeds_at_points = [spatial_interpolator(lon, lat)[0] for lon, lat in zip(lons_plants, lats_plants)]

            # Calculate production and store in reactive variable
            productions = np.abs(power_curve_normalised(wind_speeds_at_points) * capacities) # absolute values because from time to time a -0.01 MW value arises
            reactive_data.set(productions)  # Store production for download

            # Update marker pop-ups with production values
            for marker, name, capacity, operator, owner, wind_speed, production in zip(marker_cluster.markers, project_names, capacities, operators, owners, wind_speeds_at_points, productions):
                marker.popup.value = f"<strong>Project Name:</strong> {name}<br>" \
                                     f"<strong>Capacity:</strong> {capacity} MW<br>" \
                                     f"<strong>Operator:</strong> {operator}<br>" \
                                     f"<strong>Owner:</strong> {owner}<br>" \
                                     f"<strong>Wind Speed:</strong> {wind_speed:.2f} m/s<br>" \
                                     f"<strong>Production forecast:</strong> {production:.2f} MW"

            # Prepare heatmap data and update
            heatmap_data = [(lat, lon, prod) for lat, lon, prod in zip(lats_plants, lons_plants, productions)]
            for layer in m.layers:
                if isinstance(layer, Heatmap):
                    m.remove(layer)
            heatmap = Heatmap(locations=heatmap_data, radius=5, blur=5, max_zoom=10)
            m.add(heatmap)

        global initial
        # Initialize the map with the first time step
        if initial == 0:
            update_map(None)
            initial = 1

        # Observe slider value changes and update map
        slider.observe(update_map, names='value')

        # Add FullScreenControl to map
        m.add(FullScreenControl())

        return m

    # Function to download data as Excel file
    @render.download(filename="wind_production.xlsx")
    async def download_data():
        productions = reactive_data.get()  # Retrieve the latest production data

        # Create a workbook and worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Wind Power Production Data"

        # Define headers
        headers = ["Latitude", "Longitude", "Capacity (MW)", "Project Name", "Start Year", "Operator", "Owner", "Production (MW)"]
        ws.append(headers)

        # Add data for the current step
        for lat, lon, cap, name, year, operator, owner, prod in zip(lats_plants, lons_plants, capacities, project_names, start_years, operators, owners, productions):
            ws.append([lat, lon, cap, name, year, operator, owner, prod])

        # Save workbook to a bytes buffer
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)  # Set cursor to the start of the buffer

        yield buffer.getvalue()  # Return the entire buffer content

# Define absolute path to `www` folder
path_www = os.path.join(os.path.dirname(__file__), "www")

# Start the app with the `www` directory as static content
app = App(app_ui, server, static_assets={"/www": path_www})
app.run()
