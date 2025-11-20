import pandas as pd
import great_expectations as ge
from mlflow.tracking import MlflowClient
import mlflow
import os
from prefect import get_run_logger

MODEL_NAME = os.getenv("MODEL_NAME", "wine-quality-model")
MIN_MAE = float(os.getenv("MIN_MAE", "0.75"))
MAX_LOSS = float(os.getenv("MAX_LOSS", "0.80"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def check_data(csv_path: str) -> dict:
    """Vérification de la qualité des données avec Great Expectations"""
    # Récupérer le logger Prefect
    logger = get_run_logger()
    
    logger.info("=== Vérification des données avec Great Expectations ===")

    try:
        df = ge.from_pandas(pd.read_csv(csv_path))
        results = {}

        # Check 1: Pas de valeurs NaN/null
        results["no_NaN"] = df.expect_table_row_count_to_be_between(
            min_value=1, max_value=None
        ).success and all(
            df.expect_column_values_to_not_be_null(col).success 
            for col in df.columns if col != 'type'
        )

        # Check 2: Types de colonnes corrects (exclure 'type' et 'quality')
        numeric_cols = [col for col in df.columns if col not in ['type', 'quality']]
        results["no_strings_in_numeric_columns"] = all(
            df.expect_column_values_to_be_of_type(col, "float64").success
            or df.expect_column_values_to_be_of_type(col, "int64").success
            for col in numeric_cols
        )

        # Check 3: Pas de valeurs négatives (sauf pH)
        results["no_negative_values"] = all(
            df.expect_column_min_to_be_between(col, min_value=0).success
            for col in numeric_cols if col not in ["pH"]
        )

        # Check 4: pH entre 0 et 14
        results["valid_pH_range"] = df.expect_column_values_to_be_between(
            "pH", min_value=0, max_value=14
        ).success

        # Check 5: Alcohol entre 0 et 20
        results["alcohol_range"] = df.expect_column_values_to_be_between(
            "alcohol", min_value=0, max_value=20
        ).success

        # Check 6: Type de vin valide (red ou white)
        if 'type' in df.columns:
            results["valid_wine_type"] = df.expect_column_values_to_be_in_set(
                "type", value_set=["red", "white"]
            ).success

        # Check 7: Quality entre 0 et 10
        if 'quality' in df.columns:
            results["valid_quality_range"] = df.expect_column_values_to_be_between(
                "quality", min_value=0, max_value=10
            ).success

        # Logger les résultats dans Prefect
        logger.info("Résultats du data check :")
        for key, value in results.items():
            if value:
                logger.info(f"✓ {key}: OK")
            else:
                logger.error(f"✗ {key}: FAIL")

        return results
    
    except Exception as e:
        logger.error(f"Exception lors de la vérification des données: {e}", exc_info=True)
        raise  # Relancer l'exception pour que Prefect la capture


def check_model() -> dict:
    """Vérification du modèle dans MLflow"""
    # Récupérer le logger Prefect
    logger = get_run_logger()
    
    logger.info("=== Vérification du modèle dans MLflow ===")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Récupérer toutes les versions du modèle
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not versions:
            logger.warning(f"Aucun modèle '{MODEL_NAME}' trouvé.")
            return None

        # Prendre la dernière version
        latest_version = max(versions, key=lambda x: int(x.version))
        run_id = latest_version.run_id

        logger.info(f"✓ Modèle version: {latest_version.version}")
        logger.info(f"✓ Run ID: {run_id}")

        # Récupérer les métriques du run
        run = client.get_run(run_id)
        metrics = run.data.metrics

        logger.info("Métriques trouvées dans MLflow :")
        for k, v in metrics.items():
            logger.info(f" - {k}: {v:.4f}")

        results = {}
        
        # Vérifier MAE
        mae_keys = ["mean_absolute_error", "test_mae", "val_mae", "mae"]
        mae_value = None
        for key in mae_keys:
            if key in metrics:
                mae_value = metrics[key]
                break
        
        if mae_value is not None:
            results["mae_ok"] = mae_value <= MIN_MAE
            if results["mae_ok"]:
                logger.info(f"✓ MAE: {mae_value:.4f} <= {MIN_MAE} (seuil)")
            else:
                logger.error(f"✗ MAE: {mae_value:.4f} > {MIN_MAE} (seuil dépassé!)")
        else:
            results["mae_ok"] = None
            logger.warning(f"⚠ Aucune métrique MAE trouvée parmi {mae_keys}")

        # Vérifier Loss
        loss_keys = ["loss", "test_loss", "val_loss"]
        loss_value = None
        for key in loss_keys:
            if key in metrics:
                loss_value = metrics[key]
                break

        if loss_value is not None:
            results["loss_ok"] = loss_value <= MAX_LOSS
            if results["loss_ok"]:
                logger.info(f"✓ Loss: {loss_value:.4f} <= {MAX_LOSS} (seuil)")
            else:
                logger.error(f"✗ Loss: {loss_value:.4f} > {MAX_LOSS} (seuil dépassé!)")
        else:
            results["loss_ok"] = None
            logger.warning(f"⚠ Aucune métrique Loss trouvée parmi {loss_keys}")

        logger.info("Résumé du model check :")
        for k, v in results.items():
            if v is None:
                logger.warning(f" - {k}: ⚠ (métrique absente)")
            elif v:
                logger.info(f" - {k}: ✓ OK")
            else:
                logger.error(f" - {k}: ✗ FAIL")

        return results

    except Exception as e:
        logger.error(f"Exception lors de la vérification du modèle: {e}", exc_info=True)
        raise  # Relancer l'exception pour que Prefect la capture