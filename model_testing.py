import os
import glob
import pandas as pd
import pickle
from sklearn.metrics import r2_score

# Пути
BASE_DIR = os.path.dirname(__file__)
TEST_DIR = os.path.join(BASE_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def load_test_data():
    """
    Считывает все CSV из test/, возвращает X_test (матрица [[t]]), y_test.
    """
    paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.csv")))
    df_list = []
    for path in paths:
        df = pd.read_csv(path)
        df_list.append(df)
    if not df_list:
        raise FileNotFoundError("Нет файлов в папке test/")
    all_df = pd.concat(df_list, axis=0, ignore_index=True)
    X = all_df[["t"]].values.reshape(-1, 1)
    y = all_df["y_scaled"].values
    return X, y

def main():
    # 1) Загрузить модель
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2) Загрузить тестовые данные
    X_test, y_test = load_test_data()

    # 3) Предсказать
    y_pred = model.predict(X_test)

    # 4) Вычислить R^2
    score = r2_score(y_test, y_pred)

    # 5) Вывести результат
    print(f"Model test R^2 score is: {score:.3f}")

if __name__ == "__main__":
    main()
