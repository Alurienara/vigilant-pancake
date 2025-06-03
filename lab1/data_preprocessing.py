import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

def load_all_csv(folder):

    """
    Возвращает список DataFrame и названий файлов в папке.
    """

    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append((p, df))
    return dfs

def main():
    train_data = load_all_csv(TRAIN_DIR)
    all_y_train = pd.concat([df["y"] for (_, df) in train_data], axis=0).values.reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(all_y_train)

    for path, df in train_data:
        y = df["y"].values.reshape(-1, 1)
        y_scaled = scaler.transform(y).flatten()
        df["y_scaled"] = y_scaled
        df.drop(columns=["y"], inplace=True)
        df.to_csv(path, index=False)
        print(f"Предобработан и перезаписан train-файл: {path}")

    test_data = load_all_csv(TEST_DIR)
    for path, df in test_data:
        y = df["y"].values.reshape(-1, 1)
        y_scaled = scaler.transform(y).flatten()
        df["y_scaled"] = y_scaled
        df.drop(columns=["y"], inplace=True)
        df.to_csv(path, index=False)
        print(f"Предобработан и перезаписан test-файл:  {path}")

    # Для train-файлов в результате остаются колонки ['t', 'y_scaled']

if __name__ == "__main__":
    main()