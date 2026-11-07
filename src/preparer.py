"""Prepare reproducible train and test datasets."""

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
