import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

BASE_DIR   = os.path.dirname(__file__)
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(PROC_DIR, "unified_train.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

df_train = pd.read_csv(TRAIN_PATH)
X_train = df_train[["t"]].values.reshape(-1, 1)
y_train = df_train["y_scaled"].values

model = LinearRegression()
model.fit(X_train, y_train)

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)