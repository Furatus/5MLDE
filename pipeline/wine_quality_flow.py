from prefect import flow, task
import mlflow
from model_training import train_and_log_model, evaluate_model
from config import MLFLOW_URI, EXPERIMENT_NAME, DATA_PATH
from load_data import load_data
from train_test_split import train_test_split
from model_creation import create_model
from preprocessing import preprocess_data

@flow(name="Wine Quality Training Pipeline")
def wine_quality_pipeline():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    data = load_data(DATA_PATH)
    X_processed, y_processed, preprocessor = preprocess_data(data)
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X_processed, y_processed)
    num_inputs = X_train.shape[1]
    input_shape = (num_inputs, )
    model = create_model(input_shape=input_shape)
    model = train_and_log_model(model, X_train, y_train, X_val, y_val)
    evaluation = evaluate_model(model, X_test, y_test)
    
    return model

if __name__ == "__main__":
    wine_quality_pipeline.serve("Wine Quality Training Pipeline", cron="0 0/8 * * *")