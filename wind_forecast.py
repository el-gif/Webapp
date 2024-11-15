from shiny import App, ui
from ipywidgets import SelectionSlider, Layout, Play, VBox, jslink, Dropdown, HTML, Checkbox  # pip install ipywidgets==7.6.5, because version 8 has an issue with popups (https://stackoverflow.com/questions/75434737/shiny-for-python-using-add-layer-for-popus-from-ipyleaflet)
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Heatmap, WidgetControl, LayerGroup, basemaps, basemap_to_tiles, FullScreenControl, ImageOverlay, TileLayer, Marker, MarkerCluster
from ipyleaflet.velocity import Velocity
import numpy as np
import itertools
from branca.colormap import linear
import xarray as xr
from PIL import Image
import io
from base64 import b64encode
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import pandas as pd

file_path = "data/weather_forecast/data_europe.grib2"  # weather data
ds = xr.open_dataset(file_path, engine='cfgrib')
valid_times = ds['valid_time'].values

# coordinates for Europe
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# initialisation with average values
min_wind_speed = 5
max_wind_speed = 6

# extraction of wind components for current time step
lats = ds['latitude'].values
lons = ds['longitude'].values

# filtering to the selected region
lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

lats_selection = lats[lat_indices]
lons_selection = lons[lon_indices]

u_selection = []
v_selection = []

for i in range(len(ds['step'])):
    u = ds['u100'].isel(step=i).values
    v = ds['v100'].isel(step=i).values
    u_selection.append(u[np.ix_(lat_indices, lon_indices)])
    v_selection.append(v[np.ix_(lat_indices, lon_indices)])

# temporal interpolation
start_time = valid_times[0]
end_time = valid_times[-1]
total_hours = int((end_time - start_time) / np.timedelta64(1, 'h'))

step_size = np.timedelta64(1, 'h')  # interpolation step
step_size_hours = step_size / np.timedelta64(1, 'h')

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

u_selection_interpol = interpolation(u_selection)
v_selection_interpol = interpolation(v_selection)

valid_times_interpol = []
steps = int(total_hours / step_size_hours)
for i in range(steps):
    valid_times_interpol.append(start_time + i*step_size)

total_selection_interpol = []
for i in range(len(u_selection_interpol)):
    total_selection_interpol.append(np.sqrt(u_selection_interpol[i]**2 + v_selection_interpol[i]**2))
    max_wind_speed = total_selection_interpol[i].max() if total_selection_interpol[i].max() > max_wind_speed else max_wind_speed
    min_wind_speed = total_selection_interpol[i].min() if total_selection_interpol[i].min() < min_wind_speed else min_wind_speed

# colormap choice for Image Overlay
colormap = linear.viridis.scale(round(min_wind_speed), round(max_wind_speed))

# Advance preparation of HeatMap
grid_points = list(itertools.product(lats_selection, lons_selection))
latitudes, longitudes = zip(*grid_points)
heatmap_data = []
for i in range(len(total_selection_interpol)):
    heatmap_data.append([])
    for lat, lon, wind_speed in zip(latitudes, longitudes, total_selection_interpol[i].flatten()):
        heatmap_data[i].append((lat, lon, wind_speed))

initial = 0

# User interface of the Shiny App, order of elements is important!
app_ui = ui.page_fluid(
    output_widget("map")
)

# Server logic for the Shiny App
def server(input, output, session):

    # Render the Leaflet widget with render_widget, already called at initial connection
    @output
    @render_widget
    def map():

        # # Basemap dropdown options: https://ipyleaflet.readthedocs.io/en/latest/map_and_basemaps/basemaps.html, currently the one at the top is always chosen as standard
        # basemap_tiles_dict = {
        #     "Google Map": TileLayer(url='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'),
        #     "OpenStreetMap": basemap_to_tiles(basemaps.OpenStreetMap.Mapnik),
        #     "OpenTopoMap": basemap_to_tiles(basemaps.OpenTopoMap),
        #     "NASAGIBS ViirsEarthAtNight": basemap_to_tiles(basemaps.NASAGIBS.ViirsEarthAtNight2012)
        # }

        # Set initial basemap
        m = Map(center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom=5,
                layout=Layout(width='100%', height='100vh'),
                scroll_wheel_zoom=True
            )
        
        # # Load file (relative path)
        # file_path = "data/WPPs/Global-Wind-Power-Tracker-June-2024.xlsx"
        # df = pd.read_excel(file_path, sheet_name='Data')

        # # Filter the data for the geographic area in Europe
        # df_filtered = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) & 
        #                 (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

        # markers = [Marker(location=(row['Latitude'], row['Longitude'])) for _, row in df_filtered.iterrows()]
        # marker_cluster = MarkerCluster(markers=markers)

        # m.add(marker_cluster)

        # # Observe zoom changes and recalculate the marker cluster
        # def on_zoom_change(change):
        #     new_zoom = m.zoom  # The current zoom level of the map
        #     print(f"Zoom level changed to: {new_zoom}")
        #     m.remove(marker_cluster)
        #     m.add(marker_cluster)
        
        # # Observe zoom changes
        # m.observe(on_zoom_change, names='zoom')

        # # Dropdown menu to select the basemap
        # dropdown = Dropdown(
        #     options=list(basemap_tiles_dict.keys()),  # Basemap names as options
        #     description='Basemap'
        # )

        # dropdown_control = WidgetControl(widget=dropdown, position='topright')
        # m.add(dropdown_control)

        # # Activate the first basemap
        # current_basemap_layer = basemap_tiles_dict[dropdown.value]
        # m.add(current_basemap_layer)

        # # Function to update the basemap
        # def update_basemap(change):
        #     nonlocal current_basemap_layer
        #     # Remove the current basemap
        #     m.remove(current_basemap_layer)
        #     # Add the newly selected basemap
        #     current_basemap_layer = basemap_tiles_dict[dropdown.value]
        #     m.add(current_basemap_layer)

        # # Link the dropdown menu with the update function
        # dropdown.observe(update_basemap, names='value')

        # Create a slider as a WidgetControl and add it to the map
        play = Play(min=0, max=total_hours, step=step_size_hours, value=0, interval=500, description='Time Step')  # 500 ms per step
        slider = SelectionSlider(options=valid_times_interpol, value=valid_times_interpol[0], description='Time')
        jslink((play, 'value'), (slider, 'index'))
        slider_box = VBox([play, slider])

        slider_control = WidgetControl(widget=slider_box, position='topright')
        m.add(slider_control)

        layer_visibilities = {}
        heatmap_layer = None
        velocity_layer = None
        colormap_layer = None

        # Function to update the map based on the slider
        def update_map(change):

            nonlocal layer_visibilities
            nonlocal heatmap_layer, velocity_layer, colormap_layer  # to guarantee use of current layer by observe() function of checkbox
            
            time_step = slider.value
            step_index = int((time_step - start_time) / step_size)

            heatmap = Heatmap(locations=heatmap_data[step_index], radius=5, blur=5, max_zoom=10)
            
            velocity = Velocity(
                data={
                    'u': (['lat', 'lon'], u_selection_interpol[step_index]),
                    'v': (['lat', 'lon'], v_selection_interpol[step_index]),
                    'lat': lats_selection,
                    'lon': lons_selection,
                },
                zonal_speed='u',
                meridional_speed='v',
                latitude_dimension='lat',
                longitude_dimension='lon',
                velocity_scale=0.01,
                max_velocity=20,
                display_options={
                    'velocityType': 'Wind Speed',
                    'displayPosition': 'bottomleft',
                    'displayEmptyString': 'No wind data'
                }
            )

            height = len(total_selection_interpol[step_index])
            width = len(total_selection_interpol[step_index][0])

            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    hex_colour = colormap(total_selection_interpol[step_index][i][j])  # Get hex colour from colormap

                    # Convert hex colour to RGBA
                    r = int(hex_colour[1:3], 16)
                    g = int(hex_colour[3:5], 16)
                    b = int(hex_colour[5:7], 16)
                    a = 255 # Full opacity

                    rgba_image[i, j] = (r, g, b, a)

            # Convert the RGBA array to an image
            img = Image.fromarray(rgba_image, 'RGBA')          

            # Assume img is your already generated image from the RGBA array
            img_array = np.array(img)

            # Define the parameters, following section inspired from https://stackoverflow.com/questions/55955917/how-to-represent-scalar-variables-over-geographic-map-in-jupyter-notebook
            src_crs = 'EPSG:4326'
            dst_crs = 'EPSG:3857'

            # Calculate transformation parameters for EPSG:4326 -> EPSG:3857
            src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
            dst_transform, width, height = calculate_default_transform(src_crs, dst_crs, width, height, lon_min, lat_min, lon_max, lat_max)

            # Create target array for reprojection
            destination = np.zeros((4, height, width), dtype=np.uint8)

            # Reproject the image
            reproject(
                source=img_array.transpose(2, 0, 1),
                destination=destination,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.cubic
            )

            im = Image.fromarray(destination.transpose(1, 2, 0), 'RGBA')

            # Convert the transformed image to Base64
            buffer = io.BytesIO()
            im.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = b64encode(buffer.read()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"

            # Create the ImageOverlay with the transformed data
            image_overlay = ImageOverlay(
                url=img_data_url,
                bounds=[(lat_min, lon_min), (lat_max, lon_max)],  # use bounds in degrees, even if map has another CRS
                opacity=0.6
            )
            
            # Define new layers
            heatmap_layer = LayerGroup(layers=(heatmap,), name="Heatmap Layer")
            velocity_layer = LayerGroup(layers=(velocity,), name="Velocity Layer")
            colormap_layer = LayerGroup(layers=(image_overlay,), name="Colormap Layer")

            if initial == 0:
                for layer in [heatmap_layer, velocity_layer, colormap_layer]:
                    m.add(layer)
                    layer_visibilities[layer] = {'visibility': True}  # 'visibility' is set here

            # Replace old layers with new layers, without changing visibility
            if initial != 0:
                layer_visibilities_new = {}
                for layer in layer_visibilities:

                    if layer.name == "Heatmap Layer":
                        visibility = layer_visibilities[layer]['visibility']
                        if visibility:
                            m.substitute(layer, heatmap_layer)
                        layer_visibilities_new[heatmap_layer] = {'visibility': visibility}

                    elif layer.name == "Velocity Layer":
                        visibility = layer_visibilities[layer]['visibility']
                        if visibility:
                            m.substitute(layer, velocity_layer)
                        layer_visibilities_new[velocity_layer] = {'visibility': visibility}

                    elif layer.name == "Colormap Layer":
                        visibility = layer_visibilities[layer]['visibility']
                        if visibility:
                            m.substitute(layer, colormap_layer)
                        layer_visibilities_new[colormap_layer] = {'visibility': visibility}

                layer_visibilities = layer_visibilities_new

            if initial == 0:

                # Create a checkbox for each layer to control visibility
                heatmap_checkbox = Checkbox(value=True, description="Heatmap Layer")
                velocity_checkbox = Checkbox(value=True, description="Velocity Layer")
                colormap_checkbox = Checkbox(value=True, description="Colormap Layer")

                # Place the checkboxes on the map
                layer_control_box = VBox([heatmap_checkbox, velocity_checkbox, colormap_checkbox])
                layer_control_widget = WidgetControl(widget=layer_control_box, position='topright')

                # Add the custom LayerControl to the map
                m.add_control(layer_control_widget)

            # Create a function to control layer visibility
            def toggle_layer_visibility(change, layer):
                print(m.zoom)
                if change['new']:
                    # Update visibility in the dictionary
                    layer_visibilities[layer]['visibility'] = True
                    m.add(layer)  # Add layer to the map
                else:
                    # Update visibility in the dictionary
                    layer_visibilities[layer]['visibility'] = False
                    m.remove(layer)  # Remove layer from the map

            heatmap_checkbox.observe(lambda change: toggle_layer_visibility(change, heatmap_layer), names='value')  # lambda function necessary, because the callback function toggle_layer_visibility passes more than parameters
            velocity_checkbox.observe(lambda change: toggle_layer_visibility(change, velocity_layer), names='value')
            colormap_checkbox.observe(lambda change: toggle_layer_visibility(change, colormap_layer), names='value')

        # Link the slider with the update function, observe function by ipywidgets instead of reactive by shiny
        slider.observe(update_map, names='value')

        global initial
        # Initialise the map with the first time step
        if initial == 0:
            update_map(None)
            initial = 1

        # Create a custom legend as HTML
        legend_html = colormap._repr_html_()  # Generates the HTML representation of the colour scale
        # Create HTML widget
        legend_widget = HTML(value=f"""
        <div style="position: relative; z-index:9999; background-color: white; padding: 10px;">
            <h4>Wind Speed (m/s)</h4>
            {legend_html}
        </div>
        """)

        # Add WidgetControl to the map
        legend_control = WidgetControl(widget=legend_widget, position="bottomright")
        m.add(legend_control)

        m.add(FullScreenControl())

        return m

# Create and run the Shiny App
app = App(app_ui, server)

app.run()

# to run: python app.py (withou app.run() it would be shiny run app.py instead)
