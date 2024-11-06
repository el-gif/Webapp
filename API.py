from fastapi import FastAPI, HTTPException
import numpy as np
import xarray as xr
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
from scipy.interpolate import interp1d

app = FastAPI()

lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# Bestehende Daten laden (dies könnte in die Funktion integriert werden)
data_file_path = "data/weather_forecast/data_europe.grib2"
ds = xr.open_dataset(data_file_path, engine='cfgrib')

wind_speeds = np.arange(0, 25.5, 0.5)
power_output = [0]*7 + [35, 80, 155, 238, 350, 474, 630, 802, 1018, 1234, 1504, 1773, 2076, 2379, 2664, 2948, 3141, 3334, 3425, 3515, 3546, 3577, 3586, 3594, 3598, 3599] + [3600]*18
max_capacity = 3600
power_output_normalised = [value / max_capacity for value in power_output]
power_curve_normalised = interp1d(wind_speeds, power_output_normalised, kind='cubic', fill_value="extrapolate")

@app.get("/forecast")
async def get_forecast(latitude: float, longitude: float):
    if latitude < lat_min or latitude > lat_max or longitude < lon_min or longitude > lon_max:
        raise HTTPException(status_code=400, detail="Coordinates out of range")

    try:
        # Extrahiere die Windgeschwindigkeiten für alle Zeitschritte
        u_values = ds['u100'].sel(latitude=latitude, longitude=longitude, method="nearest").values
        v_values = ds['v100'].sel(latitude=latitude, longitude=longitude, method="nearest").values

        forecasts = []
        for step_index in range(len(u_values)):
            u = u_values[step_index]
            v = v_values[step_index]
            wind_speed = np.sqrt(u**2 + v**2)
            production = float(power_curve_normalised(wind_speed) * max_capacity)

            forecasts.append({
                "step": str(ds["step"][step_index].values),  # Zeitschritt als String
                "wind_speed": wind_speed,
                "production_forecast": production
            })

        forecast_data = {
            "latitude": latitude,
            "longitude": longitude,
            "forecasts": forecasts
        }
        
        return forecast_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating forecast: {str(e)}")



# FastAPI über Uvicorn starten
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
