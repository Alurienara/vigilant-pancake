import os
import pandas as pd
from catboost.datasets import titanic

"""
Скрипт для загрузки «сырых» данных Titanic:
- Используем функцию catboost.datasets.titanic() для получения исходного DataFrame.
- Сохраняем его в CSV: lab4/data/raw/titanic_raw.csv
"""

def main():
    df_raw = titanic()[0]

    out_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "titanic_raw.csv")

    df_raw.to_csv(out_path, index=False)
    print(f"Сырые данные Titanic сохранены в {out_path}")

if __name__ == "__main__":
    main()