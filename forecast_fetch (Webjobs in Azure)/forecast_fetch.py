import os
from ecmwf.opendata import Client
import datetime as dt

# Ensure the persistent directory exists
forecast_dir = "/home/data/weather_forecast/"

# DEBUG: Print current directory and target directory
print(f"Current working directory: {os.getcwd()}")
print(f"Target directory for forecast: {forecast_dir}")

# Ensure the forecast directory exists
os.makedirs(forecast_dir, exist_ok=True)

# Delete old forecast files
for old_file in os.listdir(forecast_dir):
    if old_file.startswith("forecast"):  # Filter only forecast files
        old_file_path = os.path.join(forecast_dir, old_file)  # Get full path
        if os.path.isfile(old_file_path):  # Ensure it's a file (not a folder)
            print(f"Deleting old file: {old_file}")
            os.remove(old_file_path)
            print("File deleted.")

# Determine the latest available forecast
current_utc = dt.datetime.now(dt.timezone.utc)
latest_time = 12 if current_utc.hour >= 21 else 0 if current_utc.hour >= 9 else 12

client = Client()

# Store the forecast file with a timestamp
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
target_file = os.path.join(forecast_dir, f"forecast_{timestamp}.grib")

# DEBUG: Print target file
print(f"Target file: {target_file}")

result = client.retrieve(
    type="fc",
    param=["100v", "100u"],  # U- and V-components of wind speed
    target=target_file,
    time=latest_time,
    step=list(range(0, 145, 3))
)

print(f"Forecast retrieved successfully! Saved to: {target_file}")
