# 🍷 Wine Quality MLDE

Projet MLDE déployant un modèle pour prédire la qualité du vin avec entraînement automatisé, suivi des expériences et API de prédiction.

---

## 🚀 Démarrage rapide

### 1️⃣ Lancer tout le projet
```bash
docker compose up -d
```

### 2️⃣ Accéder aux interfaces

| Service | URL | Description |
|---------|-----|-------------|
| **MLflow** | http://localhost:5000 | Suivi des expériences et modèles |
| **Prefect** | http://localhost:4200 | Orchestration des pipelines |
| **FastAPI** | http://localhost:8000 | API de prédiction |
| **Swagger** | http://localhost:8080 | Swagger de l'API |


---

## 📂 Structure du projet

```
.
├── api/
│   └── app.py                   # API FastAPI
├── dataset/
│   └── winequality.csv          # Dataset
├── mlflow_server/
│   └── Dockerfile               # Mlflow
├── pipeline/
│   ├── wine_quality_flow.py     # Pipeline principal Prefect
│   ├── model_creation.py        # Création du modèle
│   ├── model_training.py        # Entraînement du modèle
│   ├── data_quality_check.py    # Validation qualité données/modèle
│   ├── preprocessing.py         # Prétraitement des données
│   ├── load_data.py             # Chargement des données
│   ├── train_test_split.py      # Séparation des jeux de données
│   └── config.py                # Configuration
├── prefect_server/
│   └── Dockerfile               # Prefect
├── swagger.yaml                 # Swagger
└── docker-compose.yml           # Orchestration des services
```

---

## 🔄 Pipeline automatique

Le pipeline s'exécute **automatiquement toutes les 8 heures** et :

1. ✅ Valide la qualité des données
2. 📊 Charge et prétraite les données
3. 🤖 Entraîne un modèle de régression
4. 📈 Log les métriques dans MLflow
5. ✅ Valide les performances du modèle
6. 🚀 Enregistre le modèle dans MLflow

**Bonus** : Si aucun modèle n'existe au démarrage, le pipeline se lance automatiquement 

---

## 🧪 Faire une prédiction

### Via l'API
1. Aller sur http://localhost:8000
2. Tester l'endpoint `/predict` avec :

```json
{
  "type": "white",
  "fixed_acidity": 7.0,
  "volatile_acidity": 0.27,
  "citric_acid": 0.36,
  "residual_sugar": 20.7,
  "chlorides": 0.045,
  "free_sulfur_dioxide": 45.0,
  "total_sulfur_dioxide": 170.0,
  "density": 1.001,
  "pH": 3.0,
  "sulphates": 0.45,
  "alcohol": 8.8
}
```

### Via curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "white",
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 20.7,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 1.001,
    "pH": 3.0,
    "sulphates": 0.45,
    "alcohol": 8.8
  }'
```

---

## 📊 Monitoring

### MLflow (http://localhost:5000)
- 📈 Métriques : Loss, MAE
- 🏷️ Versions de modèles
- 📦 Artifacts et paramètres

### Prefect (http://localhost:4200)
- ⏰ Exécutions schedulées
- 📋 Logs détaillés
- ✅ Status des tasks

---

## ⚙️ Configuration

Les seuils de qualité sont configurables via variables d'environnement dans `docker-compose.yml` :

```yaml
environment:
  - MIN_MAE=0.75        # Seuil max pour Mean Absolute Error
  - MAX_LOSS=0.80       # Seuil max pour Loss
```

---

## 🛠️ Technologies

- **MLflow** : Suivi des expériences ML
- **Prefect** : Orchestration des workflows
- **FastAPI** : API REST pour les prédictions
- **TensorFlow/Keras** : Entraînement du modèle
- **Great Expectations** : Validation des données
- **Docker** : Containerisation

---

## 📝 Notes

- Le pipeline vérifie automatiquement les nouvelles versions de modèle
- Les checks de qualité sont non-bloquants par défaut (mode développement)
- Pour activer le mode strict, décommenter les fonctions `validate_*` dans `wine_quality_flow.py`

---

## 🎯 Objectif du projet

Démonstration d'un pipeline MLOps complet avec :
- ✅ Entraînement automatisé
- ✅ Validation de données
- ✅ Versioning de modèles
- ✅ API de production
- ✅ Monitoring centralisé

---