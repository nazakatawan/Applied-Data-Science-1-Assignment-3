
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
data_file_path = 'data.csv'
data_df = pd.read_csv(data_file_path, skiprows=4)

# Function for linear curve fitting
def linear_fit(x, a, b):
    return a * x + b

# Filter data for Urbanization and CO2 emissions
urbanization_data = data_df[data_df['Indicator Code'] == 'SP.URB.TOTL.IN.ZS']
co2_emissions_data = data_df[data_df['Indicator Code'] == 'EN.ATM.CO2E.PC']

# Select a recent year for analysis - let's use 2020
year = '2020'
urbanization_2020 = urbanization_data[['Country Name', year]].rename(columns={year: 'Urbanization'})
co2_emissions_2020 = co2_emissions_data[['Country Name', year]].rename(columns={year: 'CO2 Emissions'})

# Merging the datasets
merged_data = pd.merge(urbanization_2020, co2_emissions_2020, on='Country Name')

# Drop NaN values for clustering
merged_data_clean = merged_data.dropna()

# KMeans Clustering
kmeans = KMeans(n_clusters=3)
merged_data_clean['Cluster'] = kmeans.fit_predict(merged_data_clean[['Urbanization', 'CO2 Emissions']])

# Plotting the results with clustering
plt.figure(figsize=(12, 7))
sns.scatterplot(data=merged_data_clean, x='Urbanization', y='CO2 Emissions', hue='Cluster', palette='viridis')
plt.title('CO2 Emissions vs Urbanization with KMeans Clustering')
plt.xlabel('Urbanization (% of Total Population)')
plt.ylabel('CO2 Emissions (Metric Tons Per Capita)')
plt.grid(True)
plt.show()

# Curve Fitting for CO2 Emissions
x_data = merged_data_clean['Urbanization'].values
y_data = merged_data_clean['CO2 Emissions'].values

# Fit the data to the linear curve
params, params_covariance = curve_fit(linear_fit, x_data, y_data)

# Plotting the curve fit
plt.figure(figsize=(12, 7))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, linear_fit(x_data, *params), label='Fitted function', color='red')
plt.title('Curve Fitting for CO2 Emissions vs Urbanization')
plt.xlabel('Urbanization (% of Total Population)')
plt.ylabel('CO2 Emissions (Metric Tons Per Capita)')
plt.legend()
plt.grid(True)
plt.show()
