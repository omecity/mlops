import numpy as np
import pandas as pd
import sys
import os

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn 


def read_file(file):

    # Read the data
    df = pd.read_csv(file)

    # Get the dimensions of the DataFrame
    print(f"\nThe dimension of the original DataFrame is {df.shape}")

    # Check the first few rows 
    print(df.head())

    # Select only the columns I am interested in 
    df = df[["y", "x1", "x2"]]

    # Inspect the first few rows
    print()
    print(df.head())

    # Define features (X) and target (y)
    X = df[["x1", "x2"]]
    y = df["y"]

    return X, y



if __name__ == "__main__":


    if len(sys.argv) != 2:
        print("Usage: python mlflow_ica.py <file.csv>")
        sys.exit(1)
    
    file = sys.argv[1]

    X, y = read_file(file)


    # ---------------- Ensure experiment exists ----------------

    experiment_name = "linear_models_with_columns_x1_x2"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID {experiment_id}.")
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment '{experiment_name}' exists with ID {experiment_id}.")

    mlflow.set_experiment(experiment_name)

    # ----------------------------------------------------------

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Unwrap all the subsets into a list
    all_cols = [list(ncols) for ncol in [1, 2] for ncols in combinations(X.columns, ncol)]

    # Create an empty dictionary
    models = {}

    # Create names for the models and append into the dictionary
    for idx in all_cols:
        model = ""
        for inner_idx in range(len(idx)):
            model = model + "_" + idx[inner_idx]
        model = "model" + model
        models
        models[model] = LinearRegression()

    print()

    # Use existing requirements.txt if available
    req_file = "requirements.txt" if os.path.exists("requirements.txt") else None

    for idx, (model_name, model) in enumerate(models.items()):

        with mlflow.start_run(run_name=model_name):

            # Train the model
            model.fit(X_train[all_cols[idx]], y_train)  

            # Make predictions
            y_pred = model.predict(X_test[all_cols[idx]])

            # Evaluate the model and calculate error metrics
            mae = mean_absolute_error(y_pred, y_test)
            mse = mean_squared_error(y_pred, y_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_pred, y_test)

            # Log metrics
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)

            # Log and register model
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,              
                registered_model_name=model_name,
                pip_requirements=req_file
            )

            print(f"{model_name} logged with metrics: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    print("\nAll models are logged. Start MLflow UI with: mlflow ui")