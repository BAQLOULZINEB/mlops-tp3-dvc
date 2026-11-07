"""Train the model from the prepared training dataset."""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


with open("params.yaml", encoding="utf-8") as stream:
    params = yaml.safe_load(stream)["entrainement"]

train = pd.read_csv("data/prepare/train.csv")
features = train.drop(columns="fraude")
target = train["fraude"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=params["random_state"],
).fit(features, target)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("entrainement : modele ecrit dans models/model.pkl")
