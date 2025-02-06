import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import OneHotEncoder

# Load WPP + Production Wind JSON data
with open("data/WPPs+production+wind/WPPs+production+wind_new.json", "r", encoding="utf-8") as file:
    WPP_production_wind = json.load(file)

# Load The Wind Power data (Parquet)
df = pd.read_parquet("data/WPPs/The_Wind_Power.parquet")

# Load turbine type encoder categories
encoder = OneHotEncoder(sparse_output=False)
all_turbine_types = [wpp["Turbine"] if pd.notna(wpp["Turbine"]) else "nan" for wpp in WPP_production_wind]
encoder.fit(np.array(all_turbine_types).reshape(-1, 1))
turbine_categories = encoder.categories_[0]  # 46 categories (with nan as the 46th)

# Create a new "Other/Unknown" category for unmatched turbine types in the second dataset
df["Turbine Category"] = df["Turbine"].apply(lambda x: x if x in turbine_categories else "nan")

# Prepare features for Training dataset
features_wind = {
    "Turbine Types": [wpp["Turbine"] for wpp in WPP_production_wind],
    "Hub Heights (m)": [wpp["Hub_height"] for wpp in WPP_production_wind],
    "Capacities (MW)": [wpp["Capacity"] for wpp in WPP_production_wind],
    "Ages (months)": (pd.Timestamp("2024-12-01") - pd.to_datetime(
        [f"{wpp['Commissioning_date']}/06" if isinstance(wpp["Commissioning_date"], str) and "/" not in wpp["Commissioning_date"]
         else wpp["Commissioning_date"] for wpp in WPP_production_wind], format='%Y/%m')).days // 30,
    "Wind Speeds (m/s)": [entry[2] for wpp in WPP_production_wind for entry in wpp["Production"]]
}

# Prepare features for Forecasting dataset
features_twp = {
    "Turbine Types": df["Turbine Category"].values,
    "Hub Heights (m)": df["Hub height"].values,
    "Capacities (MW)": df["Total power"].values / 1e3,  # Convert from kW to MW
    "Ages (months)": df["Ages months"].values,
}

# Plotting function
def plot_feature_distribution(feature_name, data_wind, data_twp=None, is_discrete=False):
    plt.figure(figsize=(12, 6))
    if is_discrete:
        # Normalize frequencies within each dataset
        wind_df = pd.DataFrame({"Source": "Training data set", feature_name: data_wind})
        wind_df = wind_df[feature_name].value_counts(normalize=True).reset_index().rename(columns={"index": feature_name})
        wind_df["Source"] = "Training data set"

        twp_df = pd.DataFrame({"Source": "Forecasting data set", feature_name: data_twp}) if data_twp is not None else None
        if twp_df is not None:
            twp_df = twp_df[feature_name].value_counts(normalize=True).reset_index().rename(columns={"index": feature_name})
            twp_df["Source"] = "Forecasting data set"
            combined_data = pd.concat([wind_df, twp_df]).reset_index(drop=True)
        else:
            combined_data = wind_df

        # Bar plot with "dodge" to display bars side by side
        sns.barplot(data=combined_data, x=feature_name, y="proportion", hue="Source", dodge=True)
        plt.xticks(rotation=90)
        plt.legend(title="Dataset")
    else:
        # Histogram with very thin bars (binwidth=0.1)
        sns.histplot(data_wind, label="Training data set", bins=100, alpha=0.5, kde=False, stat="density")
        print(f"Training data set: minimal {feature_name}: {min(data_wind)}, maximal {feature_name}: {max(data_wind)}")
        if data_twp is not None:
            sns.histplot(data_twp, label="Forecasting data set", bins=100, alpha=0.5, kde=False, stat="density")
            print(f"Forecasting data set: minimal {feature_name}: {min(data_twp)}, maximal {feature_name}: {max(data_twp)}")
    
    plt.title(f"Probability Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Relative Frequency" if is_discrete else "Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot probability distributions
plot_feature_distribution("Turbine Types", features_wind["Turbine Types"], features_twp["Turbine Types"], is_discrete=True)
plot_feature_distribution("Hub Heights (m)", features_wind["Hub Heights (m)"], features_twp["Hub Heights (m)"])
plot_feature_distribution("Capacities (MW)", features_wind["Capacities (MW)"], features_twp["Capacities (MW)"])
plot_feature_distribution("Ages (months)", features_wind["Ages (months)"], features_twp["Ages (months)"])
plot_feature_distribution("Wind Speeds (m/s)", features_wind["Wind Speeds (m/s)"])