from sklearn.linear_model import LinearRegression

def train_and_predict(xs, ys):
    model = LinearRegression()
    model.fit(xs, ys)
    preds = model.predict(xs)
    return model, preds
