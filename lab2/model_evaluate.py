import os
import pandas as pd
import pickle
from sklearn.metrics import r2_score

BASE_DIR  = os.path.dirname(__file__)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
TEST_PATH = os.path.join(PROC_DIR, "unified_test.csv")

with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

df_test = pd.read_csv(TEST_PATH)
X_test = df_test[["t"]].values.reshape(-1, 1)
y_test = df_test["y_scaled"].values

score = r2_score(y_test, model.predict(X_test))
print(f"Model test R^2 score is: {score:.3f}")