import numpy as np
import pandas as pd
import xarray as xr
import json
import os
from scipy.interpolate import interp2d

# Define lead times
lead_times = list(range(0, 145, 3))

# Load wind power plant (WPP) data
with open("data/WPPs+production/WPPs+production_new_complete.json", "r", encoding="utf-8") as f:
    wpps = json.load(f)

lead_time_dicts = {str(lt): {} for lt in lead_times}  # Initialize empty structure

# Forecast file directory
forecast_dir = r"E:\MA_data\reforecast"

# Process each GRIB file (one per month-year)
grib_files = sorted([f for f in os.listdir(forecast_dir) if f.endswith(".grib")])

for grib_file in grib_files:
    print(f"Processing {grib_file}")
        
    _, year, month = grib_file.replace(".grib", "").split("_")
    year, month = int(year), int(month)
    grib_path = os.path.join(forecast_dir, grib_file)

    # Load wind speed forecast data
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib", chunks={"time": 100})
    except Exception as e:
        print(f"Error reading {grib_file}: {e} --> skipping")
        continue

    # Extract forecast timestamps and wind components
    times = pd.to_datetime(ds["valid_time"].values)
    latitudes = ds["latitude"].values
    longitudes = ds["longitude"].values
    u = ds["u100"].values
    v = ds["v100"].values

    # Process each WPP
    for i, wpp in enumerate(wpps):
        
        json_id = wpp["JSON-ID"]
        wpp_id = wpp["ID_The-Wind-Power"]
        unique_key = f"{json_id}_{wpp_id}"  # Unique identifier of a wpp

        lon = wpp["Longitude"]
        lat = wpp["Latitude"]

        # Convert production timestamps to pandas datetime format
        date_range = pd.to_datetime([entry[0] for entry in wpp["Production"]])
        production_values = [entry[1] for entry in wpp["Production"]]

        # Filter production data for this month-year
        month_mask = (date_range.year == year) & (date_range.month == month)
        production_df = pd.DataFrame({"date": date_range[month_mask], "production": np.array(production_values)[month_mask]})

        if production_df["production"].empty:
            print(f"WPP {i+1}/{len(wpps)} has no production data for {year}/{month}, skipping...")
            continue

        print(f"WPP {i+1}/{len(wpps)} for year {year}/{month}")

        forecast_data = []

        for _, row in production_df.iterrows():
            timestep = row["date"]
            production = row["production"]
            for j in range(len(times)):
                forecast_times = times[j]
                if timestep in forecast_times:
                    forecast_times = pd.DatetimeIndex(forecast_times) # convert from DatetimeArray to DatetimeIndex, because only this one has .get_loc()
                    time_index = forecast_times.get_loc(timestep)
                    lead_time = lead_times[time_index]

                    forecast_u = u[j]
                    forecast_v = v[j]

                    wind_speeds = np.sqrt(forecast_u[time_index]**2 + forecast_v[time_index]**2)
                    spatial_interpolator = interp2d(longitudes, latitudes, wind_speeds, kind='linear')
                    wind_speed_value = spatial_interpolator(lon, lat)[0]
                    wind_speed_value = round(wind_speed_value, 2)

                    forecast_data.append([lead_time, timestep, production, wind_speed_value])

        forecast_df = pd.DataFrame(forecast_data, columns=["lead_time", "forecast_time", "production", "wind_speed"])

        # Store data in structured dictionary grouped by lead time
        for lead_time in lead_times:
            lead_time_str = str(lead_time)  # Ensure keys are strings for JSON compatibility

            # Select data points corresponding to this lead time
            lead_time_mask = forecast_df["lead_time"] == lead_time
            lead_time_data = forecast_df[lead_time_mask]

            if lead_time_data.empty:
                continue  # Skip if no data for this lead time

            # Initialize new WPP entry
            lead_time_dicts[lead_time_str][unique_key] = {
                "Name": wpp["Name"],
                "Names_UK_Plants": wpp["Names_UK_Plants"],
                "Code": wpp["Code"],
                "Type": wpp["Type"],
                "Capacity": wpp["Capacity"],
                "Continent": wpp["Continent"],
                "ISO_code": wpp["ISO_code"],
                "Country": wpp["Country"],
                "State_code": wpp["State_code"],
                "Area": wpp["Area"],
                "City": wpp["City"],
                "Name_TWP": wpp["Name_TWP"],
                "Second_name": wpp["Second_name"],
                "Latitude": wpp["Latitude"],
                "Longitude": wpp["Longitude"],
                "Altitude_Depth": wpp["Altitude_Depth"],
                "Location_accuracy": wpp["Location_accuracy"],
                "Shore_distance_km": wpp["Shore_distance_km"],
                "Manufacturer": wpp["Manufacturer"],
                "Turbine": wpp["Turbine"],
                "Hub_height": wpp["Hub_height"],
                "Number_of_turbines": wpp["Number_of_turbines"],
                "Developer": wpp["Developer"],
                "Operator": wpp["Operator"],
                "Owner": wpp["Owner"],
                "Commissioning_date": wpp["Commissioning_date"],
                "Status": wpp["Status"],
                "Decommissioning_date": wpp["Decommissioning_date"],
                "Source_link": wpp["Source_link"],
                "Last_update": wpp["Last_update"],
                "Time Series": [[str(date), production, wind_speed] for _, date, production, wind_speed in lead_time_data[["lead_time", "forecast_time", "production", "wind_speed"]].values.tolist()]
            }

    output_json_file = os.path.join(forecast_dir, f"{grib_file.replace('.grib', '_wind.json')}")

    # Save updated JSON
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(lead_time_dicts, f, indent=4)

    print(f"Updated JSON file saved: {output_json_file}")
