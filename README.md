# TP3 - DVC : versionner les donnees et rendre le pipeline reproductible

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull          # necessite l'acces au stockage DVC (../dvc-stockage)
dvc repro
dvc metrics show
```

Serveur MLflow (pour `src/journaliser.py`) :

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Tags : `v1` (5000 lignes, F1 0.8636), `v2` (7000 lignes, F1 0.8423).
