import os
import pandas as pd
from prefect import flow, task

@flow
def load_data(file_path):
    check_file_exists(file_path)
    data = pd.read_csv(file_path)
    return data

@task
def check_file_exists(file_path):
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    return True