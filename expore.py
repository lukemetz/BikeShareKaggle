import pandas as pd
import numpy as np

train_path = "/home/luke/datasets/bikeShare/train.csv"

base_features = ["datetime", "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed"]
target_features = ["casual", "registered", "count"]

def get_features():
    data = pd.read_csv(train_path)
    features = np.vstack([data[f].values for f in base_features]).transpose()
    targets = np.vstack([data[f].values for f in target_features]).transpose()

    return features, targets

def main():
    features, targets = get_features()
    Y = targets[:, target_features.index("count")]

if __name__ == "__main__":
    main()
