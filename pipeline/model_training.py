import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from config import MODEL_PARAMS, MODEL_NAME
from prefect import task

# FONCTION CLASSIQUE = Logique métier pure
def train_model_core(model, X_train, y_train, X_val, y_val, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size']):
    """Entraînement du modèle (fonction interne)"""
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

# FONCTION CLASSIQUE = Logique métier pure
def evaluate_model_core(model, X_test, y_test):
    """Évaluation du modèle (fonction interne)"""
    evaluation = model.evaluate(X_test, y_test, verbose=1)
    return evaluation

# TASK PREFECT = Orchestration + appel de la logique
@task
def train_and_log_model(model, X_train, y_train, X_val, y_val, epochs=MODEL_PARAMS['epochs'], batch_size=MODEL_PARAMS['batch_size']):
    """Task Prefect principal : entraîne et log le modèle"""
    with mlflow.start_run():
        # Appel des fonctions normales (pas des tasks)
        model, history = train_model_core(model, X_train, y_train, X_val, y_val, epochs, batch_size)
        
        loss = evaluate_model_core(model, X_val, y_val)
        
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("val_mae", history.history.get('val_mae', [0])[-1])

        sample_input = X_val[:100]
        sample_predictions = model.predict(sample_input)

        signature = infer_signature(sample_input, sample_predictions)
        
        model_info = mlflow.keras.log_model(
            model=model,
            artifact_path=MODEL_NAME,
            signature=signature,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "tensorflow",
                "keras",
                "numpy",
                "pandas",
                "scikit-learn"
            ]
        )

        print(f"✓ Modèle loggé et enregistré: {MODEL_NAME}")
        print(f"✓ Version: {model_info.registered_model_version}")
        print(f"Validation loss: {loss}")
        
        return model, model_info

@task
def evaluate_model(model, X_test, y_test):
    """Task Prefect : évalue le modèle sur le test set"""
    evaluation = evaluate_model_core(model, X_test, y_test)
    return evaluation