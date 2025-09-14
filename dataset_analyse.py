import pandas as pd

# Load your dataset
df = pd.read_csv('synthetic_flipkart_user_dataset.csv')

# View dataset shape
print("Shape of dataset:", df.shape)

# List all column names
print("Column names:", df.columns.tolist())

# View data types of each column
print("\nData Types:\n", df.dtypes)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Get basic statistics
print("\nBasic Statistics:\n", df.describe())