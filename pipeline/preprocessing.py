from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from prefect import flow, task

@flow
def preprocess_data(data):
    """Pipeline principal de prétraitement des données"""
    X, y = prepare_X_y(data)
    
    cat_cols = ["type"]
    num_cols = list(X.columns[1:])
    
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    X_processed = preprocessor.fit_transform(X)
    y_processed = prepare_y(y)

    return X_processed, y_processed, preprocessor

@task
def prepare_y(y):
    """Normaliser la target (qualité) entre 0 et 1"""
    return y/10

@task 
def create_num_scaler():
    """Créer le pipeline pour les colonnes numériques : imputation + normalisation"""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")), 
        ("scaler", MinMaxScaler())  
    ])

@task
def create_cat_encoder():
    """Créer le pipeline pour les colonnes catégorielles : imputation + encoding"""
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")), 
        ("encoder", OneHotEncoder(drop="first", sparse_output=False))  
    ])


@flow
def create_preprocessor(numeric_features, categorical_features):
    """Créer le transformateur global qui applique les pipelines selon le type de colonne"""
    num_scaler = create_num_scaler()
    cat_encoder = create_cat_encoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_scaler, numeric_features),
            ("categorical", cat_encoder, categorical_features),
        ]
    )
    return preprocessor

@task
def prepare_X_y(data):
    """Séparer les features (X) de la target (y)"""
    target = "quality"
    X = data.drop(target, axis=1) 
    y = data[target]  
    return X, y