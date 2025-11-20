DATA_PATH = "/app/winequality.csv"

MLFLOW_URI = "http://mlflow:5000" 

EXPERIMENT_NAME = "wine-quality-prediction"

RUN_NAME = "wine-quality-run"

MODEL_NAME = "wine-quality-model"

PREFECT_API_URL = "http://prefect:4200/api"

MODEL_PARAMS = {
    "epochs": 10,
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "mse",
    "dropout_rate": 0.2,
    "hidden_units": [512, 256],
}