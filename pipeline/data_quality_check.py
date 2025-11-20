import pandas as pd
import great_expectations as ge
from mlflow.tracking import MlflowClient
import mlflow
import os

MODEL_NAME = os.getenv("MODEL_NAME", "wine-quality-model")
MIN_MAE = float(os.getenv("MIN_MAE", "0.75"))
MAX_LOSS = float(os.getenv("MAX_LOSS", "0.80"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def check_data(csv_path: str) -> dict:
    """ Vérification de la qualité des données avec Great Expectations """
    print("\n=== Vérification des données avec Great Expectations ===\n")

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

    print("Résultats du data check :")
    for key, value in results.items():
        status = '✓ OK' if value else '✗ FAIL'
        print(f" - {key}: {status}")

    return results


def check_model() -> dict:
    """ Vérification du modèle dans MLflow """
    print("\n=== Vérification du modèle dans MLflow ===\n")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Récupérer toutes les versions du modèle
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        
        if not versions:
            print(f"⚠ Aucun modèle '{MODEL_NAME}' trouvé.")
            return None

        # Prendre la dernière version
        latest_version = max(versions, key=lambda x: int(x.version))
        run_id = latest_version.run_id

        print(f"✓ Modèle version: {latest_version.version}")
        print(f"✓ Run ID: {run_id}")

        # Récupérer les métriques du run
        run = client.get_run(run_id)
        metrics = run.data.metrics

        print("\nMétriques trouvées dans MLflow :")
        for k, v in metrics.items():
            print(f" - {k}: {v:.4f}")

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
            print(f"\n✓ MAE trouvé: {mae_value:.4f} (seuil: {MIN_MAE})")
        else:
            results["mae_ok"] = None
            print(f"\n⚠ Aucune métrique MAE trouvée dans {mae_keys}")

        # Vérifier Loss
        loss_keys = ["loss", "test_loss", "val_loss"]
        loss_value = None
        for key in loss_keys:
            if key in metrics:
                loss_value = metrics[key]
                break

        if loss_value is not None:
            results["loss_ok"] = loss_value <= MAX_LOSS
            print(f"✓ Loss trouvé: {loss_value:.4f} (seuil: {MAX_LOSS})")
        else:
            results["loss_ok"] = None
            print(f"⚠ Aucune métrique Loss trouvée dans {loss_keys}")

        print("\nRésultats du model check :")
        for k, v in results.items():
            if v is None:
                print(f" - {k}: ⚠ (métrique absente)")
            elif v:
                print(f" - {k}: ✓ OK")
            else:
                print(f" - {k}: ✗ FAIL")

        return results

    except Exception as e:
        print(f"✗ Erreur lors de la vérification du modèle: {e}")
        return None