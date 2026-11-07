"""TP3 - src/preparer.py

ROLE      : Etape 1 du pipeline : decoupe les donnees en apprentissage (75 %) et test (25 %).
ENTREE    : data/transactions.csv + params.yaml (section preparation)
SORTIE    : data/prepare/train.csv et data/prepare/test.csv
A RETENIR : Dans le TP3, c'est dvc.yaml qui lance ce script (dvc repro), seulement si ses dependances changent.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


with open("params.yaml", encoding="utf-8") as stream:
    params = yaml.safe_load(stream)["preparation"]

dataframe = pd.read_csv("data/transactions.csv")
train, test = train_test_split(
    dataframe,
    test_size=params["test_size"],
    random_state=params["random_state"],
)

Path("data/prepare").mkdir(parents=True, exist_ok=True)
train.to_csv("data/prepare/train.csv", index=False)
test.to_csv("data/prepare/test.csv", index=False)
print("preparation :", len(train), "lignes train /", len(test), "lignes test")
