import os
import re
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer


def descriptive_stats(dataset_name, df):
    print(f"\n Descriptive statistics for dataset: {dataset_name}")
    print(f"Shape: {df.shape}")
    print(df.head())
    print("\nSummary (describe):")
    print(df.describe(include="all"))
    print("\nMissing values count:")
    print(df.isnull().sum())


iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
descriptive_stats("Iris", iris_df)
breast_cancer = load_breast_cancer()
bc_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
descriptive_stats("Breast Cancer", bc_df)
with open("cleveland.data", "r", encoding="latin1", errors="ignore") as f:
    txt = f.read()
tokens = re.findall(r"-?\d+(?:\.\d+)?|\?", txt)
nrows = len(tokens) // 14
tokens = tokens[: nrows * 14]
rows = [tokens[i : i + 14] for i in range(0, len(tokens), 14)]
cols = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
heart_df = pd.DataFrame(rows, columns=cols)
for c in cols:
    heart_df[c] = pd.to_numeric(heart_df[c].replace("?", pd.NA), errors="coerce")
descriptive_stats("Heart Disease", heart_df)
genes_df = pd.read_csv("genes.csv")
descriptive_stats("GenesData", genes_df)
