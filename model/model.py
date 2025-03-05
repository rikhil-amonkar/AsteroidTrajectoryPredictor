from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import math

# Import all impact and pha data for space objects
impact_data = pd.read_csv("/Users/rikhilamacpro/VS Projects/Asteroid Trajectory Predictor ML Project #7/AsteroidTrajectoryPredictor/impact_data/cneos_fireball_data.csv")
neo_close = pd.read_csv("/Users/rikhilamacpro/VS Projects/Asteroid Trajectory Predictor ML Project #7/AsteroidTrajectoryPredictor/impact_data/neo_close_approaches.csv")

# Change sign of coordinate depending on direction
def direction_to_decimal(coord):
    direction = coord[-1]
    value_coord = float(coord[:-1])
    if direction in ["S", "W"]:
        value_coord = -value_coord  
    return value_coord

def average_neo_diameter(diameter_range):
    if diameter_range:
        split_diameter = diameter_range.split()
        calculated_diameter = (float(split_diameter[3]) + float(split_diameter[0])) / 2
        average_diameter = round(calculated_diameter, 2)
        return average_diameter
    return None

# Convert data into dataframe
df_impact = pd.DataFrame(impact_data)
df_neo = pd.DataFrame(neo_close)

# Remove rows with any NaN values for impact data
df_cleaned_impact = df_impact.dropna()
df_cleaned_neo = df_neo.dropna()

# Process longitude and latitude data columns for impact data
df_cleaned_impact['Latitude (deg.)'] = df_cleaned_impact['Latitude (deg.)'].apply(direction_to_decimal)
df_cleaned_impact['Longitude (deg.)'] = df_cleaned_impact['Longitude (deg.)'].apply(direction_to_decimal)

# ****** NEEDED FORMULAS ******

# Energy(J) = (0.5) * (mass) * (velocity)^2
# Diameter(km) (est.) = [ 1329 / sqrt(albedo(Pv)) ] * (10)^(-0.2 * H(mag))
# Density(kg/m^3) = (mass) / (volume)
# Volume(km/s) (spherical) = [ (4/3) * pi ] * (diameter/2)^3
# Mass(kg) (est.) = (density) * ( (4/3) * pi ] * ((diameter * 1000)/2)^3 ] )

# Process longitude and latitude data columns for impact data
df_cleaned_neo['Diameter'] = df_cleaned_neo['Diameter'].apply(average_neo_diameter)

# Calculate estimated volume of each asteroid and add to list
neo_volume = []
for diameter in df_cleaned_neo['Diameter']:
    if diameter != None:
        diameter = float(diameter)
        vol_of_neo = (((4/3) * math.pi) * ((diameter / 2)**3))
        neo_volume.append(vol_of_neo)
    
print(neo_volume)


# Process 







