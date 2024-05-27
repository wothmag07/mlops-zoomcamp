
import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient


'''MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
client.create_experiment("nyc-taxi-duration")
'''


# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-pred")

# Enable autologging for scikit-learn models
mlflow.sklearn.autolog()

# Function to load a pickled object from a file
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Command-line interface for training and logging
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # Start an MLflow run for logging
    with mlflow.start_run():

        # Load training and validation data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Set hyperparameters (e.g., max_depth) and log them
        depth = 10
        mlflow.log_param("max_depth", depth)

        # Train a RandomForestRegressor model
        rf = RandomForestRegressor(max_depth=depth, random_state=0)
        rf.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = rf.predict(X_val)

        # Calculate and log root mean squared error (RMSE)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Log the trained model
        mlflow.sklearn.log_model(rf, artifact_path="models")

        # Print the default artifacts URI
        print(f"Default Artifacts URI: '{mlflow.get_artifact_uri()}'")


if __name__ == '__main__':
    run_train()
