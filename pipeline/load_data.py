import os
import pandas as pd
from prefect import flow, task
from config import DL_TEMP_FILENAME

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

@task
def download_data(url: str):
    try:
        df = pd.read_csv(url)
        df.to_csv(DL_TEMP_FILENAME, index=False)
        return DL_TEMP_FILENAME
    except Exception as e:
        raise RuntimeError(f"Erreur lors du téléchargement des données: {e}")
    
@task 
def delete_temp_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        return True
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la suppression du fichier temporaire: {e}")