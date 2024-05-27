import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_error

# Set up the MLflow tracking server URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename: str):
    """
    Load a pickled object from a file.
    """
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    """
    Run hyperparameter optimization using Hyperopt to find the best Random Forest model.
    The results are logged using MLflow.
    """
    # Load training and validation data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        """
        Objective function for Hyperopt. Trains a Random Forest model with the given parameters,
        evaluates it on the validation set, and logs the results to MLflow.
        """
        with mlflow.start_run():
            # Initialize and train the model
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            # Predict on the validation set
            y_pred = rf.predict(X_val)
            # Calculate RMSE
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            # Log parameters and metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            return {'loss': rmse, 'status': STATUS_OK}

    # Define the search space for hyperparameter optimization
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42  # Ensure reproducibility
    }

    # Set random state for reproducible results
    rstate = np.random.default_rng(42)

    # Run the optimization using the Tree-structured Parzen Estimator (TPE) algorithm
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':
    run_optimization()
