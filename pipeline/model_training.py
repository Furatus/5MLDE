import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import keras
from config import MODEL_PARAMS, MODEL_NAME
from prefect import flow, task

@task
def train_model(model, X_train, y_train, X_val, y_val, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size']):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

@task
def evaluate_model(model, X_test, y_test):
    evaluation = model.evaluate(X_test, y_test, verbose=1)
    return evaluation

@task
def train_and_log_model(model, X_train, y_train, X_val, y_val, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size']):
    with mlflow.start_run():
        model, history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)
        
        loss = evaluate_model(model, X_val, y_val)
        
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("val_mae", history.history['val_mse'][-1])

        sample_input = X_val[:100]
        sample_predictions = model.predict(sample_input)

        # Infer signature from sample data
        signature = infer_signature(sample_input, sample_predictions)
        mlflow.keras.log_model(model, MODEL_NAME, signature=signature)

        print(f"Validation loss: {loss}")
        
        return model