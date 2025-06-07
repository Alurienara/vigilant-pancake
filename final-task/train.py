import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)

X = df.select_dtypes(include=['number']).drop(columns=['Survived'], errors='ignore')
y = df['Survived'] if 'Survived' in df.columns else df.iloc[:, -1]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "wb") as f:
    pickle.dump(model, f)