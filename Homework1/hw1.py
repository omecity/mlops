import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# Read CSV file into pandas
df = pd.read_csv("sampregdata.csv")



# Take a look at the first few rows
print("\nOriginal DataFrame: \n")
print(df.head())
print()



# Get the dimensions of the DataFrame
print(f"\nThe dimension of the original DataFrame is {df.shape}")



# Take at the columns 
print("\nThe name of columns of the DataFrame:")
print(df.columns)



# Remove the first column by its index df.columns[0] gets the name of the 
# first column axis=1 specifies that a column is being dropped inplace=True
# modifies the DataFrame directly
df.drop(columns=df.columns[0], inplace=True)


# Get the dimensions of the DataFrame
print(f"\nThe dimension of the DataFrame after first column has been removed {df.shape}")



print("\nDataFrame after removing the first column: \n")
print(df.head())
print()