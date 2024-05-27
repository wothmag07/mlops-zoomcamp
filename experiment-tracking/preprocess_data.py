import os
import pickle
import click
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename: str):
    """Save the given object to a file using pickle."""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    """
    Read the Parquet file and convert it into a DataFrame.
    Calculate the duration of trips in minutes and filter out trips
    that are less than 1 minute or more than 60 minutes.
    Convert categorical columns to strings.
    """
    df = pd.read_parquet(filename)

    # Calculate trip duration in minutes
    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    # Filter out trips outside the duration range
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    """
    Preprocess the DataFrame by creating a new column 'PU_DO' combining
    'PULocationID' and 'DOLocationID'. Convert the DataFrame into a format
    suitable for the DictVectorizer and then transform it.
    """
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    # Convert the DataFrame to a list of dictionaries
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    # Fit and transform the DictVectorizer if needed
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    """
    Main function to prepare the data. Load the raw Parquet files,
    preprocess them, and save the processed data and the DictVectorizer.
    """
    # Load Parquet files for training, validation, and test sets
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
    )

    # Extract the target variable
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess the data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create destination path folder if it does not exist
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and preprocessed datasets to files
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()
