from data_generator import generate_clean_data, generate_noisy_data
from train_model import train_and_predict
from sklearn.metrics import mean_squared_error

def test_clean_data_quality():
    xs, ys = generate_clean_data()
    _, preds = train_and_predict(xs, ys)
    mse = mean_squared_error(ys, preds)
    assert mse < 1.0

def test_noisy_data_degrades_quality():
    xs_clean, ys_clean = generate_clean_data()
    xs_noisy, ys_noisy = generate_noisy_data()

    _, preds_clean = train_and_predict(xs_clean, ys_clean)
    _, preds_noisy = train_and_predict(xs_noisy, ys_noisy)

    mse_clean = mean_squared_error(ys_clean, preds_clean)
    mse_noisy = mean_squared_error(ys_noisy, preds_noisy)

    assert mse_noisy > mse_clean
