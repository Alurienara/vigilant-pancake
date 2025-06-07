import os
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    raw_path = os.path.join(base_dir, "data", "raw", "titanic_raw.csv")
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    filled_path = os.path.join(proc_dir, "titanic_filled.csv")

    df = pd.read_csv(raw_path)

    imputer = SimpleImputer(strategy="median")
    df["Age"] = imputer.fit_transform(df[["Age"]])

    df.to_csv(filled_path, index=False)
    print(f"Filled missing Age values → saved to {filled_path}")

if __name__ == "__main__":
    main()