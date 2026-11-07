"""TP3 - src/evaluer.py

ROLE      : Etape 3 du pipeline : mesure le modele sur le jeu de test.
ENTREE    : data/prepare/test.csv + models/model.pkl
SORTIE    : reports/metrics.json (f1, precision, recall) + reports/confusion.png
A RETENIR : metrics.json est declare 'cache: false' : il reste dans Git pour etre compare (dvc metrics diff).
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


test = pd.read_csv("data/prepare/test.csv")
features = test.drop(columns="fraude")
target = test["fraude"]
model = joblib.load("models/model.pkl")
predictions = model.predict(features)

metrics = {
    "f1": round(f1_score(target, predictions), 4),
    "precision": round(precision_score(target, predictions), 4),
    "recall": round(recall_score(target, predictions), 4),
}

Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w", encoding="utf-8") as stream:
    json.dump(metrics, stream, indent=2)
ConfusionMatrixDisplay.from_predictions(target, predictions)
plt.savefig("reports/confusion.png", dpi=110, bbox_inches="tight")
plt.close()
print("evaluation :", metrics)
