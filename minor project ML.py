import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Load the Crop Production dataset 
crop_production = pd.read_csv("crop_production_data.csv")

# Display the first few rows of the dataset
print(crop_production.head())
# Group data by year and calculate the total production
yearly_production = crop_production.groupby('year')['production'].sum()

# Plot a line chart to visualize production trends over time
plt.figure(figsize=(10, 6))
plt.plot(yearly_production.index, yearly_production.values, marker='o', linestyle='-', color='b')
plt.title("Crop Production Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Total Production")
plt.grid(True)
plt.show()
# Calculate the total production for each crop type
crop_production_by_crop = crop_production.groupby('crop')['production'].sum()

# Sort and visualize the top and bottom crops by production
top_crops = crop_production_by_crop.nlargest(10)
bottom_crops = crop_production_by_crop.nsmallest(10)

# Create bar plots for top and bottom crops
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
top_crops.plot(kind='bar', color='g')
plt.title("Top 10 Crops by Production")

plt.subplot(1, 2, 2)
bottom_crops.plot(kind='bar', color='r')
plt.title("Bottom 10 Crops by Production")

plt.tight_layout()
plt.show()
# Group data by crop and season and calculate the average production
seasonal_production = crop_production.groupby(['crop', 'season'])['production'].mean().unstack()

# Create a heatmap to visualize seasonal variation
plt.figure(figsize=(10, 6))
sns.heatmap(seasonal_production, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("Seasonal Variation in Crop Production")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.show()
# Calculate the total production for each state
state_production = crop_production.groupby('state')['production'].sum()

# Sort and visualize the top producing states
top_states = state_production.nlargest(10)

# Create a bar plot for top producing states
plt.figure(figsize=(10, 6))
top_states.plot(kind='bar', color='b')
plt.title("Top 10 Crop Producing States")
plt.xlabel("State")
plt.ylabel("Total Production")
plt.xticks(rotation=45)
plt.show()
# Calculate crop yields for each crop
crop_yield = crop_production.groupby('crop')['yield'].mean().sort_values(ascending=False)

# Visualize the top and bottom crops by yield
plt.figure(figsize=(10, 6))
crop_yield.head(10).plot(kind='bar', color='g')
plt.title("Top 10 Crops by Yield")
plt.xlabel("Crop")
plt.ylabel("Yield")
plt.show()
