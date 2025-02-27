from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Import all impact and pha data for space objects
impact_data = pd.read_csv("/Users/rikhilamacpro/VS Projects/Astroid Trajectory Predictor ML Project #7/AstroidTrajectoryPredictor/impact_data/cneos_fireball_data.csv")
pred_asteroids_data = pd.read_csv("/Users/rikhilamacpro/VS Projects/Astroid Trajectory Predictor ML Project #7/AstroidTrajectoryPredictor/impact_data/cneos_sentry_summary_data.csv")
# potential_haz_data = pd.read_json("/Users/rikhilamacpro/VS Projects/Astroid Trajectory Predictor ML Project #7/AstroidTrajectoryPredictor/impact_data/pha_extended.json")

# Change sign of coordinate depending on direction
def convert_to_decimal(coord):
    direction = coord[-1]
    value_coord = float(coord[:-1])
    if direction in ["S", "W"]:
        value_coord = -value_coord  
    return value_coord

# Change joules of energy to kilojoules
def energy_to_kj(energy):
    kj_energy = energy / 1000
    return kj_energy

# Convert data into dataframe
df_impact = pd.DataFrame(impact_data)
df_predicted = pd.DataFrame(pred_asteroids_data)

# Remove rows with any NaN values
df_cleaned_impact = df_impact.dropna()
df_cleaned_predicted = df_predicted.dropna()

print(df_cleaned_predicted)

# Process long and lat data columns
df_cleaned_impact['Latitude (deg.)'] = df_cleaned_impact['Latitude (deg.)'].apply(convert_to_decimal)
df_cleaned_impact['Longitude (deg.)'] = df_cleaned_impact['Longitude (deg.)'].apply(convert_to_decimal)

# Process radiated energy data from J into kJ for easier display
df_cleaned_impact['Total Radiated Energy (J)'] = df_cleaned_impact['Total Radiated Energy (J)'].apply(energy_to_kj)
df_cleaned_impact = df_cleaned_impact.rename(columns={'Total Radiated Energy (J)': 'Total Radiated Energy (kJ)'})

print(df_cleaned_impact.head())
print(df_cleaned_predicted.head())

