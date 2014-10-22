import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split

train_path = "/home/luke/datasets/bikeShare/train.csv"
test_path = "/home/luke/datasets/bikeShare/test.csv"

base_features = [
    #"datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed"]
target_features = ["casual", "registered", "count"]

def get_data():
    data = pd.read_csv(train_path)
    return data

def get_features(data):
    features = np.vstack([data[f].values for f in base_features]).transpose()
    return features

def get_targets(data):
    targets = np.vstack([data[f].values for f in target_features]).transpose()
    return targets

def shuffle(a, b):
    p = np.random.permutation(a.shape[0])
    return a[p, :], b[p, :]

def main():
    data = get_data()
    features = get_features(data)
    targets = get_targets(data)

    features, targets = shuffle(features, targets)

    Y = targets[:, target_features.index("count")]

    train_x, test_x, train_y, test_y = train_test_split(features, Y, test_size=0.33, random_state=3)

    model = Ridge()
    model.fit(train_x, train_y)
    predicted_y = model.predict(test_x)

    print predicted_y
    predicted_y = np.clip(predicted_y, 0, 1e100)
    print score(predicted_y, test_y)
    export_test(model, "out.csv")

def export_test(model, output):
    data = pd.read_csv(test_path)
    features = get_features(data)
    result = model.predict(features)
    result = np.clip(result, 0, 1e100)

    out = np.vstack([data["datetime"].values, result]).transpose()
    data["count"] = result
    data.to_csv(output, cols=["datetime", "count"], index=False)

def score(predicted, true):
    inner = (np.log(predicted + 1) - np.log(true+1))
    return np.sqrt(np.mean(inner * inner))

if __name__ == "__main__":
    main()
