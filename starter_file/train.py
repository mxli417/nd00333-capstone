import os
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


# Main variables
dataset_name = "heart_failure_kaggle"
target_column = "DEATH_EVENT"
primary_metric_name = "accuracy"


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate for fitting the model")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of estimators for the GBM model")
    args = parser.parse_args()

    # log hyperparameters
    run = Run.get_context()
    run.log("learning_rate:", np.float(args.learning_rate))
    run.log("n_estimators:", np.int(args.n_estimators))

    # get dataset
    url = "https://raw.githubusercontent.com/mxli417/nd00333-capstone/60c04db4bff6057f4ae90ba350a048a7a9e69f48/data/heart_failure_clinical_records_dataset.csv"
    heartfail_data = pd.read_csv(url)
    
    # generate a stratified train/test split to account for data imbalance
    X = heartfail_data.drop(columns=target_column)  # matrix
    y = heartfail_data[target_column]  # vector
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # fit the model
    model = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
    ).fit(X_train, y_train)

    # record the accuracy
    accuracy = model.score(X_test, y_test)
    run.log(primary_metric_name, np.float(accuracy))

    # dump the model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(value=model, './outputs/model.joblib')


if __name__ == '__main__':
    main()
