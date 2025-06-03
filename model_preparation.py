import os
import glob
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Пути
BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "train")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def load_train_data():
    """
    Считывает все CSV из train/, возвращает X_train (матрица [[t]]), y_train (вектор).
    """
    paths = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    df_list = []
    for path in paths:
        df = pd.read_csv(path)
        df_list.append(df)
    if not df_list:
        raise FileNotFoundError("Нет файлов в папке train/")
    all_df = pd.concat(df_list, axis=0, ignore_index=True)
    X = all_df[["t"]].values.reshape(-1, 1)
    y = all_df["y_scaled"].values
    return X, y

def main():
    X_train, y_train = load_train_data()

    # 1) Инициализируем и обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 2) Сохраняем её через pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель обучена и сохранена в: {MODEL_PATH}")

if __name__ == "__main__":
    main()
