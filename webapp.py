import io
import os
from openpyxl import Workbook
from shiny import ui, App, reactive
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Marker, MarkerCluster, WidgetControl, FullScreenControl, Heatmap, AwesomeIcon
from ipywidgets import SelectionSlider, Play, VBox, jslink, Layout, HTML
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d, interp2d # pip install scipy==1.13.1, interp2d much faster than RegularGridInterpolator, even if depracated
import base64
from ecmwf.opendata import Client
import time

# Define initial variables
initial = 0
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# Pfad zu den gespeicherten Wetterdaten
data_file_path = "data/data_europe.grib2"
# Festgelegtes Zeitintervall für Aktualisierungen (in Sekunden)
time_threshold = 24 * 3600  # 24 Stunden in Sekunden
overwrite = 0 # force overwriting of data

# Funktion, um das Alter der Datei zu überprüfen
def is_data_stale(file_path, time_threshold):
    if not os.path.exists(file_path):
        # Datei existiert nicht, Daten müssen abgerufen werden
        return True
    # Alter der Datei ermitteln (in Sekunden seit der letzten Änderung)
    file_age = time.time() - os.path.getmtime(file_path)
    # Überprüfen, ob die Datei älter als der festgelegte Schwellenwert ist
    return file_age > time_threshold

# ECMWF-Client initialisieren
client = Client(
    source="ecmwf",
    model="ifs",
    resol="0p25"
)

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

if is_data_stale(data_file_path, time_threshold) or overwrite:
    print("Daten sind veraltet, existieren nicht oder sollen überschrieben werden. Abruf neuer Daten.")
    # Abrufen der API-Daten, da sie entweder fehlen oder älter als 10 Stunden sind
    result = client.retrieve( # retrieve data worldwide, because no area tag available (https://github.com/ecmwf/ecmwf-opendata, https://www.ecmwf.int/en/forecasts/datasets/open-data)
        type="fc",  
        param=["100v", "100u"],  # U- und V-Komponenten der Windgeschwindigkeit
        target=data_file_path,
        time=0,  # Vorhersagezeit (Modelllauf um 00z)
        step=step_selection
    )
    print("Neue Daten wurden erfolgreich abgerufen und gespeichert.")
else:
    print("Daten sind aktuell und werden verwendet.")

# Load the WPPs (Excel file)
WPP_file = "data/Global-Wind-Power-Tracker-June-2024.xlsx"
df = pd.read_excel(WPP_file, sheet_name='Data')

# Filter the data for Europe and extract relevant columns
df_filtered = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) & (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]
lats_plants = df_filtered['Latitude'].values
lons_plants = df_filtered['Longitude'].values
capacities = df_filtered['Capacity (MW)'].values
project_names = df_filtered['Project Name'].values
statuses = df_filtered['Status'].values
operators = df_filtered['Operator'].values
owners = df_filtered['Owner'].values

# Load the wind data (Grib2 file)
wind_file = "data/data_europe.grib2"
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

# server-Funktion
def server(input, output, session):

    # Funktion für den Download der Produktionsdaten (mit Konvertierung von datetime64 zu datetime)
    async def download_data_for_marker(plant_index):
        
        lat_plant = lats_plants[plant_index]
        lon_plant = lons_plants[plant_index]

        productions = []
        for step_index in range(len(valid_times_interpol)):
            spatial_interpolator = interp2d(lons, lats, total_selection_interpol[step_index], kind='cubic')
            wind_speeds_at_points = spatial_interpolator(lon_plant, lat_plant)[0]
            productions.append(np.abs(power_curve_normalised(wind_speeds_at_points) * capacities[plant_index])) # absolute values because from time to time a -0.01 MW value arises

        # Workbook erstellen
        wb = Workbook()
        ws = wb.active
        # Titel auf maximal 31 Zeichen kürzen, um Warning zu vermeiden
        ws.title = f"Prod Data {project_names[plant_index][:25]}"

        # Kopfzeilen
        headers = ["Time Step", "Wind speed (m/s)", "Production (MW)"]
        ws.append(headers)

        # Produktionsdaten pro Zeitschritt hinzufügen
        for time, production in zip(valid_times_interpol, productions):
            # Konvertiere numpy.datetime64 zu datetime
            time_as_datetime = pd.to_datetime(time).to_pydatetime()
            ws.append([time_as_datetime, production, wind_speeds_at_points])

        # Speichern in Buffer und zurückgeben
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

        markers = []
        for index, (lat, lon, name, capacity, status, operator, owner) in enumerate(zip(lats_plants, lons_plants, project_names, capacities, statuses, operators, owners)):
            # Choose the color based on status
            if status == "operating":
                color = "green"
            elif status in ["construction", "pre-construction", "announced"]:
                color = "orange"
            else:
                color = "red"

            # Create an AwesomeIcon for the marker
            icon = AwesomeIcon(
                name="circle",
                marker_color=color
            )

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
                icon=icon,
                rise_on_hover=True
            )
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
        # Initialize the map with the first time step
        if initial == 0:
            update_map(None)
            initial = 1

        # Observe slider value changes and update map
        slider.observe(update_map, names='value')

        # Add FullScreenControl to map
        m.add(FullScreenControl())

        return m

    # Funktion zum Codieren in Base64 und Senden der Nachricht
    @reactive.Effect
    @reactive.event(input.selected_plant_index)
    async def trigger_download():
        plant_index = input.selected_plant_index()
        # Wenn ein Plant Index ausgewählt wurde, dann lade die Datei herunter
        if plant_index is not None:
            # Datei als bytes abrufen
            file_data = await download_data_for_marker(plant_index)
            # Datei in Base64 codieren
            file_data_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Nachricht an den Client senden mit Base64-codierter JSON-Datei
            await session.send_custom_message("download", {
                "data": file_data_base64,
                "filename": f"production_{project_names[plant_index]}.xlsx"
            })

# Define absolute path to `www` folder
path_www = os.path.join(os.path.dirname(__file__), "www")

# Start the app with the `www` directory as static content
app = App(app_ui, server, static_assets={"/www": path_www})
app.run()