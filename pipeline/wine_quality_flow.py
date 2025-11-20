from prefect import flow, task
import mlflow
from mlflow.tracking import MlflowClient
from model_training import train_and_log_model, evaluate_model
from config import MLFLOW_URI, EXPERIMENT_NAME, DATA_PATH, MODEL_NAME, DL_TEMP_FILENAME
from load_data import load_data
from train_test_split import train_test_split
from model_creation import create_model
from preprocessing import preprocess_data
from data_quality_check import check_data, check_model
from prefect import get_run_logger
from load_data import check_file_exists, download_data, delete_temp_file

@task
def validate_input_data(data_path: str):
    """Vérifier la qualité des données AVANT l'entraînement - VERSION STRICTE (mais non-bloquante)"""
    logger = get_run_logger()
    logger.info("Validation stricte des données d'entrée...")
    try:
        results = check_data(data_path)
        
        if not all(results.values()):
            failed_checks = [k for k, v in results.items() if not v]
            logger.error(f"ERREUR: Data quality checks échoués: {failed_checks}")
            logger.warning("Le pipeline continue pour analyse, mais ces erreurs doivent être corrigées!")
        else:
            logger.info("Données validées avec succès!")
        
        return results
    except Exception as e:
        logger.error(f"ERREUR: Exception lors de la validation des données: {e}")
        logger.warning("Le pipeline continue pour analyse...")
        return None

@task
def validate_trained_model():
    """Vérifier la qualité du modèle APRÈS l'entraînement - VERSION STRICTE (mais non-bloquante)"""
    logger = get_run_logger()
    logger.info("Validation stricte du modèle entraîné...")
    try:
        results = check_model()
        
        if results is None:
            logger.error("ERREUR: Impossible de récupérer les métriques du modèle")
            logger.warning("Le pipeline continue pour analyse...")
            return None
        
        if not all(v for v in results.values() if v is not None):
            failed_checks = [k for k, v in results.items() if v is False]
            logger.error(f"ERREUR: Model quality checks échoués: {failed_checks}")
            logger.warning("Le pipeline continue pour analyse, mais ces erreurs doivent être corrigées!")
        else:
            logger.info("Modèle validé avec succès!")
        
        return results
    except Exception as e:
        logger.error(f"ERREUR: Exception lors de la validation du modèle: {e}")
        logger.warning("Le pipeline continue pour analyse...")
        return None

@task
def soft_validate_input_data(data_path: str):
    """Vérifier la qualité des données AVANT l'entraînement - VERSION SOUPLE"""
    logger = get_run_logger()
    logger.info("Validation souple des données d'entrée...")
    try:
        results = check_data(data_path)
        
        if not all(results.values()):
            failed_checks = [k for k, v in results.items() if not v]
            logger.warning(f"WARNING: Data quality checks échoués: {failed_checks}")
            logger.warning("Le pipeline continue malgré les erreurs...")
        else:
            logger.info("Données validées avec succès!")
        
        return results
    except Exception as e:
        logger.warning(f"WARNING: Erreur lors de la validation des données: {e}")
        logger.warning("Le pipeline continue malgré l'erreur...")
        return None

@task
def soft_validate_trained_model():
    """Vérifier la qualité du modèle APRÈS l'entraînement - VERSION SOUPLE"""
    logger = get_run_logger()
    logger.info("Validation souple du modèle entraîné...")
    try:
        results = check_model()
        
        if results is None:
            logger.warning("WARNING: Impossible de récupérer les métriques du modèle")
            logger.warning("Le pipeline continue malgré l'erreur...")
            return None
        
        if not all(v for v in results.values() if v is not None):
            failed_checks = [k for k, v in results.items() if v is False]
            logger.warning(f"WARNING: Model quality checks échoués: {failed_checks}")
            logger.warning("Le pipeline continue malgré les erreurs...")
        else:
            logger.info("Modèle validé avec succès!")
        
        return results
    except Exception as e:
        logger.warning(f"WARNING: Erreur lors de la validation du modèle: {e}")
        logger.warning("Le pipeline continue malgré l'erreur...")
        return None


def check_model_exists() -> bool:
    """Vérifier si un modèle existe déjà dans MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        return len(versions) > 0
    except Exception as e:
        print(f"Erreur lors de la vérification du modèle: {e}")
        return False


@flow(name="Wine Quality Training Pipeline")
def wine_quality_pipeline(data_url: str = "", DATA_PATH: str = DATA_PATH):
    """Pipeline d'entraînement du modèle Wine Quality"""
    try:
        logger = get_run_logger()
    
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Téléchargement des données si une URL est fournie
        if data_url != "":
            logger.info(f"Téléchargement des données depuis l'URL: {data_url}")
            DATA_PATH = download_data(data_url)
            logger.info(f"Données téléchargées et sauvegardées dans: {DATA_PATH}")

        check_file_exists(DATA_PATH)
    
        # Validation des données
        # Version SOUPLE :
    
        soft_validate_input_data(DATA_PATH)
    
        # Version STRICTE :
        # validate_input_data(DATA_PATH)
    
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
        logger.info(f"Évaluation finale - Test loss: {evaluation}")
    
        # Validation du modèle
        # Version SOUPLE :
        soft_validate_trained_model()
    
        # Version STRICTE :
        # validate_trained_model()
    
        logger.info("Pipeline complété avec succès!")
        return model, model_info
    finally:
        if data_url:
            delete_temp_file(DL_TEMP_FILENAME)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("VÉRIFICATION DE L'EXISTENCE D'UN MODÈLE")
    print("="*60)
    
    model_exists = check_model_exists()
    
    if model_exists:
        print(f"Modèle '{MODEL_NAME}' trouvé dans MLflow")
        print("Le pipeline s'exécutera selon le schedule (toutes les 8h)")
    else:
        print(f"Aucun modèle '{MODEL_NAME}' trouvé dans MLflow")
        print("Lancement immédiat du premier entraînement...")
        
        # Lancer immédiatement le pipeline
        try:
            wine_quality_pipeline()
            print("\nPremier entraînement terminé avec succès!")
        except Exception as e:
            print(f"\nErreur lors du premier entraînement: {e}")
            sys.exit(1)
    
    # Démarrer le serveur avec le schedule
    print(f"\nDémarrage du schedule (cron: 0 0/8 * * *)")
    wine_quality_pipeline.serve(
        "Wine Quality Training Pipeline", 
        cron="0 0/8 * * *"
    )