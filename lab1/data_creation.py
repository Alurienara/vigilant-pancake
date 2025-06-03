import os
import numpy as np
import pandas as pd

# Параметры генерации
NUM_SAMPLES = 200  # длина одного ряда
TRAIN_FILES = [
    ("normal", 0.1, False),      # обычный синус с небольшим шумом
    ("noisy", 0.5, False),       # синус с большим шумом
    ("anomalies", 0.2, True)     # синус с шумом + выбросы
]
TEST_FILES = [
    ("normal_test", 0.1, False),
    ("noisy_test", 0.5, True)
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "train")
OUTPUT_DIR_TEST = os.path.join(os.path.dirname(__file__), "test")

def make_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

def generate_series(length, noise_level, with_anomalies):
    """
    Генерирует синтетический временной ряд:
    y = sin(2π * t / T) + шум (+ опционально выбросы)
    """
    t = np.arange(length)
    T = 50  # период синуса
    base = np.sin(2 * np.pi * t / T)
    noise = np.random.normal(scale=noise_level, size=length)
    series = base + noise

    if with_anomalies:
        # добавим несколько выбросов (+5 или −5) в случайных точках
        num_anoms = max(1, length // 50)
        indices = np.random.choice(length, size=num_anoms, replace=False)
        for idx in indices:
            series[idx] += np.random.choice([5, -5])
    return series

def save_series(series, prefix, folder):
    """
    Сохраняет один набор в формате CSV: первый столбец — индекс t, второй — значение y.
    """
    df = pd.DataFrame({
        "t": np.arange(len(series)),
        "y": series
    })
    path = os.path.join(folder, f"{prefix}.csv")
    df.to_csv(path, index=False)
    print(f"Сгенерирован и сохранён файл: {path}")

def main():
    make_dirs()

    # Генерация train-файлов
    for name, noise, anom in TRAIN_FILES:
        arr = generate_series(NUM_SAMPLES, noise, anom)
        save_series(arr, name, OUTPUT_DIR)

    # Генерация test-файлов
    for name, noise, anom in TEST_FILES:
        arr = generate_series(NUM_SAMPLES, noise, anom)
        save_series(arr, name, OUTPUT_DIR_TEST)

if __name__ == "__main__":
    main()
