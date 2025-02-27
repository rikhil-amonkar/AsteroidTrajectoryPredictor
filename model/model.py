from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Import all impact and pha data for space objects
impact_data = pd.read_csv("/Users/rikhilamacpro/VS Projects/Astroid Trajectory Predictor ML Project #7/AstroidTrajectoryPredictor/impact_data/cneos_fireball_data.csv")
potential_haz_data = pd.read_json("/Users/rikhilamacpro/VS Projects/Astroid Trajectory Predictor ML Project #7/AstroidTrajectoryPredictor/impact_data/pha_extended.json")

print(impact_data.head())
print(potential_haz_data.head())

# Change sign of coordinate depending on direction
def convert_to_decimal(coord):
    direction = coord[-1]
    value_coord = float(coord[:-1])

    if direction in ["S", "W"]:
        value_coord = -value_coord  

    return value_coord

# Example usage:
latitude = []
longitude = []

# Convert to dataframe
df = pd.DataFrame(impact_data)

# Remove rows with any NaN values
df_cleaned = df.dropna()

for lat in df_cleaned['Latitude (deg.)']:
    print(lat)

print(latitude, longitude)    

# Process long and lat data columns
# df['Latitude (deg.)'] = df['Latitude (deg.)'].apply(convert_to_decimal)
# df['Longitude'] = df['Longitude'].apply(convert_to_decimal)