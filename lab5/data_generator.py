import numpy as np

def generate_clean_data(seed=42):
    np.random.seed(seed)
    xs = np.linspace(0, 10, 100)
    ys = xs + np.random.random(100) * 2 - 1
    return xs.reshape(-1, 1), ys

def generate_noisy_data(seed=42):
    np.random.seed(seed)
    xs = np.linspace(0, 10, 100)
    ys = xs + np.random.random(100) * 2 - 1
    ys[25:45] *= 2
    return xs.reshape(-1, 1), ys
