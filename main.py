
import pandas as pd

# Load the student dataset
data = pd.read_csv("dataset.csv")

# Display the first 5 student records
print("STUDENT PERFORMANCE DATASET")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display statistical information
print("\nStatistical Summary:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)
