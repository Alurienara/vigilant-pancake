import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import sys
import json

model_path = sys.argv[1]
data_path = sys.argv[2]

with open(model_path, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(data_path)

X = df.select_dtypes(include=['number']).drop(columns=['Survived'], errors='ignore')
y = df['Survived'] if 'Survived' in df.columns else df.iloc[:, -1]

preds = model.predict(X)
acc = accuracy_score(y, preds)

print(f"Accuracy: {acc:.4f}")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)