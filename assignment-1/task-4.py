import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np

DATA_FILE = "data/cows.csv"
CATEGORICAL_COLUMNS = ['name', 'race', 'character', 'music']


def print_distances(original_data, metric):
    data = original_data
    data = data.drop(columns=CATEGORICAL_COLUMNS)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    distances = pairwise_distances(data, metric=metric)
    n_neighbors = 2

    for i in range(len(distances)):
        name = original_data.iloc[[i]].loc[:, 'name'].to_numpy()[0]
        neighbors_index = distances[i].argsort()[::-1][-n_neighbors - 1:][::-1][1:]
        neighbors_names = original_data.iloc[neighbors_index].loc[:, 'name'].to_numpy()

        print(
            f"{i} - {name} -> closest {n_neighbors} neighbours: {neighbors_names[0]} {np.round(distances[i][neighbors_index[0]], 3)} | {neighbors_names[1]} {np.round(distances[i][neighbors_index[1]], 3)}")


def main():
    data = pd.read_csv(DATA_FILE)

    ## Exercise a
    print("-- Euclidean distance --")
    print_distances(data, "euclidean")
    print("-- Mahalanobis distance --")
    print_distances(data, "mahalanobis")


if __name__ == '__main__':
    main()
