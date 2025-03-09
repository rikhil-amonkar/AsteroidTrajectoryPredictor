from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

# **************** NEEDED FORMULAS ****************

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

# **************** DATA PROCESSING ****************

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

def kilotons_to_joules(energy):
    return energy * 4.184e12

# Rename and add all column names and values to match similar features
df_cleaned_neo['Estimated Mass (kg)'] = neo_mass # Estimated mass
df_cleaned_neo['Estimated Volume (m^3)'] = neo_volume # Estimated volume
df_cleaned_neo['Calculated Total Impact Energy (J)'] = neo_poss_imp_energy # Estimated impact energy
df_cleaned_impact['Diameter (m)'] = impact_diam_est # Estimated diameter
df_cleaned_impact['Estimated Mass (kg)'] = impact_mass_est # Estimated mass
df_cleaned_neo = df_cleaned_neo.rename(columns={'V relative(km/s)': 'Velocity (km/s)'})

# Convert all impact energy values from kilotons to joules
df_cleaned_impact['Calculated Total Impact Energy (kt)'] = df_cleaned_impact['Calculated Total Impact Energy (kt)'].apply(kilotons_to_joules)
df_cleaned_impact = df_cleaned_impact.rename(columns={'Calculated Total Impact Energy (kt)': 'Calculated Total Impact Energy (J)'})

# Add labels to NEO data (label = 0 for NEOs since we don't know impact)
df_cleaned_neo['Impact_Label'] = 0

# Add labels to impact data (label = 1 for impacted asteroids since we know impact)
df_cleaned_neo['Impact_Label'] = 1

# Combine both dataframes into one
df_combined_data = pd.concat([df_cleaned_neo, df_cleaned_impact], ignore_index=True)
df_combined_data = df_combined_data.fillna(0)  # or df_combined_data.fillna(df_combined_data.mean())

# **************** MACHINE LEARNING MODEL ****************

# Select features to be used for training
common_features = ["Velocity (km/s)", "Diameter (m)", "Estimated Mass (kg)", "Calculated Total Impact Energy (J)"]

# Select features and labels
X = df_combined_data[common_features]
y = df_combined_data['Impact_Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Create a linear regression model
model = LogisticRegression(class_weight="balanced")

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions using the testing data
predictions = model.predict(X_test_scaled)

# Convert continuous predictions to binary labels using threshold
predictions_binary = (predictions > 0.5).astype(int)

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions_binary)
print(f"\nThis model is {(accuracy * 100):.2f}% accurate.")

# **************** USER PROBABILITY PREDICITONS ****************

if __name__ == "__main__":  # User input for asteroid features
    print("\nPlease enter the following features for the asteroid: ")
    
    # User inputs only velocity, mass, and diameter
    velocity = float(input("Velocity (km/s): "))
    mass = float(input("Estimated Mass (kg): "))
    diameter = float(input("Diameter (m): "))
    
    # Calculate the total impact energy (in Joules)
    impact_energy = 0.5 * mass * (velocity * 1000) ** 2  # converting velocity to m/s
    
    # Print calculated impact energy
    print(f"Estimated Total Impact Energy (J): {impact_energy:.2e}")
    
    # Scale user input
    user_input = [[velocity, diameter, mass, impact_energy]]
    user_input_scaled = X_scaler.transform(user_input)

    # Make prediction (get probability for both classes)
    user_prediction_proba = model.predict_proba(user_input_scaled)

    # Extract the probability of the asteroid hitting Earth (class 1)
    hit_probability = user_prediction_proba[0][1]

    print(f"\nThe probability that this asteroid will hit Earth is: {hit_probability * 100:.2f}%")

# **************** GRAPH OF IMPACT ****************

# Constants
k = 1.5  # Rock impact material constant

# Calculate crater radius using empirical relation
crater_radius = (k * (diameter ** 0.78) * (velocity ** 0.44) * (mass ** 0.12)) / 1000 # in km

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-crater_radius * 1.2, crater_radius * 1.2)
ax.set_ylim(-crater_radius * 1.2, crater_radius * 1.2)
ax.set_aspect('equal', 'box')

# Initial crater (small circle)
crater = plt.Circle((0, 0), 0.1, color='brown', fill=True)
ax.add_patch(crater)
ax.set_facecolor('black')

# Function to update crater size in animation
def update(frame):
    radius = (frame / 50) * crater_radius  # Expand over time
    crater.set_radius(radius)
    crater.set_alpha(1 - (frame / 50))  # Fades out over time

# Animate
ani = animation.FuncAnimation(fig, update, frames=50, interval=50)

# Display
plt.title(f"Crater Formation (Estimated Impact Radius: {crater_radius:.2f} km)")
plt.grid(True)
plt.show()










