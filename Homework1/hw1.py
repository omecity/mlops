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



# Take a look at the first few rows
print("\nThe DataFrame info after removing the first column: \n")
print(df.info())



# Define features (X) and target (y)
# Replace 'target_column' with the actual column you want to predict
X = df.drop(columns=["y"])
y = df["y"]



# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Fit separate models with one variable at a time

def build_linreg(X, y, n_cols):

    results = {}
    
    for combo in combinations(X.columns, n_cols):

        # create a regression model
        model = LinearRegression()

        # train the model
        model.fit(X_train[list(combo)], y_train)  

        # make predictions
        y_pred = model.predict(X_test[list(combo)])

        # evaluate the model
        mse_value = mean_squared_error(y_pred, y_test)

        results[combo] = mse_value
        if len(list(combo)) == 1:
            print(f"MSE with {combo[0]}: {mse_value:.4f}")
        else:
            print(f"MSE with {combo}: {mse_value:.4f}")

    return model, results