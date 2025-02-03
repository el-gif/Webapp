import numpy as np
import pandas as pd
import xarray as xr
import json
import os

# Define lead times
lead_times = list(range(0, 91)) + [93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144]

# Load wind power plant (WPP) data
with open("data/WPPs+production_new.json", "r", encoding="utf-8") as f:
    wpps = json.load(f)

# Initialize dictionary for structured forecast data
lead_time_dicts = {lt: {} for lt in lead_times}

# Forecast file directory
forecast_dir = "reforecast_data"

# Process each WPP
for index, wpp in enumerate(wpps):
    json_id = wpp["JSON-ID"]
    wpp_id = wpp["ID_The-Wind-Power"]

    # Convert production timestamps to pandas datetime format
    date_range = pd.to_datetime([entry[0] for entry in wpp["Production"]])
    production_values = [entry[1] for entry in wpp["Production"]]

    # Create a DataFrame for production data
    production_df = pd.DataFrame({"date": date_range, "production": production_values})

    # Process monthly forecast files
    for date in date_range:
        year, month = date.year, date.month
        grib_file = os.path.join("model3", forecast_dir, f"{index}_{json_id}_{year}_{month:02d}.grib")

        # Skip if no forecast file exists
        if not os.path.exists(grib_file):
            print(f"Skipping missing forecast: {grib_file}")
            continue

        # Load wind speed forecast data
        ds = xr.open_dataset(grib_file, engine="cfgrib")

        # Compute wind speed from u100 and v100 wind components
        u100 = ds["u100"].values
        v100 = ds["v100"].values
        wind_speeds = np.sqrt(u100**2 + v100**2)

        # Convert forecast times to pandas datetime format
        forecast_times = pd.to_datetime(ds.valid_time.values)

        # Create a DataFrame for wind speed forecasts
        forecast_df = pd.DataFrame({
            "lead_time": np.tile(lead_times, len(forecast_times)),
            "forecast_time": np.concatenate(forecast_times),
            "wind_speed": wind_speeds.flatten()
        })

        # Sort forecasts by forecast_time
        forecast_df = forecast_df.sort_values(by="forecast_time").reset_index(drop=True)

        # Merge production and forecast data based on matching timestamps
        merged_df = pd.merge(production_df, forecast_df, left_on="date", right_on="forecast_time", how="inner")

        # Skip if no matching timestamps found
        if merged_df.empty:
            print(f"No matching data for WPP {wpp['Name']} in {year}-{month:02d}, skipping...")
            continue

        # Store data in structured dictionary grouped by lead time
        for lead_time, group in merged_df.groupby("lead_time"):
            if json_id not in lead_time_dicts[lead_time]:
                lead_time_dicts[lead_time][json_id] = {
                    "Name": wpp["Name"],
                    "Names_UK_Plants": wpp["Names_UK_Plants"],
                    "ID_The-Wind-Power": wpp["ID_The-Wind-Power"],
                    "JSON-ID": wpp["JSON-ID"],
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
                    "Production": []
                }

            # Append production and wind speed data for this lead time and WPP
            lead_time_dicts[lead_time][json_id]["Production"].extend([
                [str(date), production, wind_speed]
                for date, production, wind_speed in group[["date", "production", "wind_speed"]].values.tolist()
            ])

# Save to JSON
output_json_file = "forecast_production_data.json"
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(lead_time_dicts, f, indent=4)

print(f"Saved JSON file: {output_json_file}")
