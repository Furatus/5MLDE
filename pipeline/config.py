DATA_PATH = "/app/winequality.csv"

MLFLOW_URI = "http://mlflow:5000" 

EXPERIMENT_NAME = "wine-quality-prediction"

RUN_NAME = "wine-quality-run"

MODEL_NAME = "wine-quality-model"

PREFECT_API_URL = "http://prefect:4200/api"

DL_TEMP_FILENAME = "./winequality_temp.csv"

MODEL_PARAMS = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "mse",
    "dropout_rate": 0.2,
    "hidden_units": [512, 256],
}