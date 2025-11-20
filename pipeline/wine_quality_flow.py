from prefect import flow, task
import mlflow
from model_training import train_and_log_model, evaluate_model
from config import MLFLOW_URI, EXPERIMENT_NAME, DATA_PATH
from load_data import load_data
from train_test_split import train_test_split
from model_creation import create_model
from preprocessing import preprocess_data
from data_quality_check import check_data, check_model

@task
def validate_input_data(data_path: str):
    """Vérifier la qualité des données AVANT l'entraînement - VERSION BLOQUANTE"""
    print("\n🔍 Validation stricte des données d'entrée...")
    results = check_data(data_path)
    
    if not all(results.values()):
        failed_checks = [k for k, v in results.items() if not v]
        raise ValueError(f"❌ PIPELINE ARRÊTÉ: Data quality checks échoués: {failed_checks}")
    
    print("✅ Données validées avec succès!")
    return results

@task
def validate_trained_model():
    """Vérifier la qualité du modèle APRÈS l'entraînement - VERSION BLOQUANTE"""
    print("\n🔍 Validation stricte du modèle entraîné...")
    results = check_model()
    
    if results is None:
        raise ValueError("❌ PIPELINE ARRÊTÉ: Impossible de récupérer les métriques du modèle")
    
    if not all(v for v in results.values() if v is not None):
        failed_checks = [k for k, v in results.items() if v is False]
        raise ValueError(f"❌ PIPELINE ARRÊTÉ: Model quality checks échoués: {failed_checks}")
    
    print("✅ Modèle validé avec succès!")
    return results


@task
def soft_validate_input_data(data_path: str):
    """Vérifier la qualité des données AVANT l'entraînement - VERSION NON-BLOQUANTE"""
    print("\n🔍 Validation des données d'entrée (non-bloquante)...")
    try:
        results = check_data(data_path)
        
        if not all(results.values()):
            failed_checks = [k for k, v in results.items() if not v]
            print(f"⚠️ WARNING: Data quality checks échoués: {failed_checks}")
            print("⚠️ Le pipeline continue malgré les erreurs...")
        else:
            print("✅ Données validées avec succès!")
        
        return results
    except Exception as e:
        print(f"⚠️ WARNING: Erreur lors de la validation des données: {e}")
        print("⚠️ Le pipeline continue malgré l'erreur...")
        return None

@task
def soft_validate_trained_model():
    """Vérifier la qualité du modèle APRÈS l'entraînement - VERSION NON-BLOQUANTE"""
    print("\n🔍 Validation du modèle entraîné (non-bloquante)...")
    try:
        results = check_model()
        
        if results is None:
            print("⚠️ WARNING: Impossible de récupérer les métriques du modèle")
            print("⚠️ Le pipeline continue malgré l'erreur...")
            return None
        
        if not all(v for v in results.values() if v is not None):
            failed_checks = [k for k, v in results.items() if v is False]
            print(f"⚠️ WARNING: Model quality checks échoués: {failed_checks}")
            print("⚠️ Le pipeline continue malgré les erreurs...")
        else:
            print("✅ Modèle validé avec succès!")
        
        return results
    except Exception as e:
        print(f"⚠️ WARNING: Erreur lors de la validation du modèle: {e}")
        print("⚠️ Le pipeline continue malgré l'erreur...")
        return None

@flow(name="Wine Quality Training Pipeline")
def wine_quality_pipeline():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Validation des données
    
    # Version BLOQUANTE :
    # validate_input_data(DATA_PATH)
    
    # Version NON-BLOQUANTE :
    soft_validate_input_data(DATA_PATH)
    
    # Chargement et préparation des données
    data = load_data(DATA_PATH)
    X_processed, y_processed, preprocessor = preprocess_data(data)
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split(X_processed, y_processed)
    
    # Création et entraînement du modèle
    num_inputs = X_train.shape[1]
    input_shape = (num_inputs, )
    model = create_model(input_shape=input_shape)
    model, model_info = train_and_log_model(model, X_train, y_train, X_val, y_val)
    
    # Évaluation
    evaluation = evaluate_model(model, X_test, y_test)
    print(f"✓ Évaluation finale - Test loss: {evaluation}")
    
    # Validation du modèle
    
    # Version BLOQUANTE :
    # validate_trained_model()
    
    # Version NON-BLOQUANTE :
    soft_validate_trained_model()
    
    print("\n🎉 Pipeline complété avec succès!")
    return model, model_info


if __name__ == "__main__":
    wine_quality_pipeline.serve("Wine Quality Training Pipeline", cron="0 0/8 * * *")