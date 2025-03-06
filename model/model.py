from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import math

# ****** NEEDED FORMULAS ******

# Energy(J) = (0.5) * (mass) * (velocity)^2
# Diameter(km) (est.) = [ 1329 / sqrt(albedo(Pv)) ] * (10)^(-0.2 * H(mag))
# Diameter (km) = ((3 * mass) / (4 * pi * density)) ^ (1/3) / 1000
# Volume(km/s) (spherical) = [ (4/3) * pi ] * (diameter/2)^3
# Density(kg/m^3) = (mass) / (volume)
# Mass(kg) (est.) = (density) * ( (4/3) * pi ] * ((diameter * 1000)/2)^3 ] )
# Impact Energy(J) = kt * 4.184e12
# Impact Mass Est.(kg) = (2 * E) / (v ** 2) 

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

# Average density of asteroids for most calculations
avg_density_of_neo = 3000 # kg/m^3

# Process longitude and latitude data columns for impact data
df_cleaned_impact['Latitude (deg.)'] = df_cleaned_impact['Latitude (deg.)'].apply(direction_to_decimal)
df_cleaned_impact['Longitude (deg.)'] = df_cleaned_impact['Longitude (deg.)'].apply(direction_to_decimal)

# Calculate estimated mass for all impacted asteroids and add to list
impact_mass_est = [] # in kilograms (kg)
for energy, velocity in zip(df_cleaned_impact['Calculated Total Impact Energy (kt)'], df_cleaned_impact['Velocity (km/s)']):
    imp_energy = float(energy * 4.184e12)
    imp_mass = (2 * imp_energy) / (velocity ** 2) 
    impact_mass_est.append(imp_mass)

# Calculate estimated diameter for all impacted asteroids and add to list
impact_diam_est = [] # in meters (m)
for mass in impact_mass_est:
    imp_diameter = round(((3 * mass) / (4 * math.pi * avg_density_of_neo))**(1/3), 2)
    impact_diam_est.append(imp_diameter)

# Process longitude and latitude data columns for impact data
df_cleaned_neo['Diameter'] = df_cleaned_neo['Diameter'].apply(average_neo_diameter)
df_cleaned_neo = df_cleaned_neo.rename(columns={'Diameter': 'Diameter (m)'})

# Calculate estimated volume of each asteroid and add to list
neo_volume = [] # in cubic meters (m^3)
for diameter in df_cleaned_neo['Diameter (m)']:
    if diameter != None:
        diameter = float(diameter)
        vol_of_neo = round((((4/3) * math.pi) * ((diameter / 2)**3)), 2)
        neo_volume.append(vol_of_neo)

# Calculate estimated mass of each asteroid and add to list
neo_mass = [] # in kilograms (kg())
for diameter in df_cleaned_neo['Diameter (m)']:
    if diameter != None:
        diameter = float(diameter)
        mass_of_neo = round(((avg_density_of_neo) * ((4/3) *  math.pi) * (((diameter * 1000) / 2)**3)), 2)
        neo_mass.append(mass_of_neo)

# Calculate possible impact energy of near earth objects and add to list
neo_poss_imp_energy = [] # in joules (J)
for mass, velocity in zip(neo_mass, df_cleaned_neo['V relative(km/s)']):
    neo_imp_energy_est = (mass) * (velocity)**2
    neo_poss_imp_energy.append(neo_imp_energy_est)
    
# print(neo_volume)
# print("\n\n")
# print(neo_mass)
print("\n\n")
print(df_cleaned_neo.columns)
print("\n\n")
print(df_cleaned_impact.columns)
# print("\n\n")
# print(df_cleaned_neo.head())


# Process 







