import os
import pandas as pd

def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    inp = os.path.join(base, "data", "processed", "titanic_filled.csv")
    out_dir = os.path.join(base, "data", "processed")
    out = os.path.join(out_dir, "titanic_ohe.csv")

    df = pd.read_csv(inp)
    df = pd.get_dummies(df, columns=["Sex"], prefix="Sex")
    df.to_csv(out, index=False)
    print(f"One-Hot Encoding done → saved to {out}")

if __name__ == "__main__":
    main()