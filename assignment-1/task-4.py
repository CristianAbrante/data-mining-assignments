import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist

DATA_FILE = "data/cows.csv"
CATEGORICAL_COLUMNS = ['name', 'race', 'character', 'music']


def print_distances(original_data, distances, similarity=False):
    n_neighbors = 2

    for i in range(len(distances)):
        name = original_data.iloc[[i]].loc[:, 'name'].to_numpy()[0]
        sorted_array = distances[i].argsort()
        sorted_array = sorted_array if similarity else sorted_array[::-1]

        neighbors_index = sorted_array[-n_neighbors - 1:][::-1][1:]
        neighbors_names = original_data.iloc[neighbors_index].loc[:, 'name'].to_numpy()

        print(
            f"{i} - {name} -> closest {n_neighbors} neighbours: {neighbors_names[0]} {np.round(distances[i][neighbors_index[0]], 3)} | {neighbors_names[1]} {np.round(distances[i][neighbors_index[1]], 3)}")


def calculate_probabilities(data):
    goodal_probs = {}
    for column in data:
        # print(data[column])
        column_array = data[column].to_numpy()
        for element in column_array:
            if goodal_probs.get(element) == None:
                goodal_probs[element] = 1
            else:
                goodal_probs[element] = goodal_probs[element] + 1
    for key in goodal_probs:
        goodal_probs[key] = goodal_probs[key] / data.shape[0]

    return goodal_probs


def goodall_distance(probs, features_a, features_b):
    intersection = np.intersect1d(features_a, features_b)
    goodall_score = 0.0

    for elem in intersection:
        goodall_score += (1.0 - np.power(probs[elem], 2))

    return goodall_score / len(features_a)


def main():
    data = pd.read_csv(DATA_FILE)

    ## Exercise a
    data_for_calculations = data.drop(columns=CATEGORICAL_COLUMNS)
    scaler = MinMaxScaler()
    scaler.fit(data_for_calculations)
    data_for_calculations = scaler.transform(data_for_calculations)

    print("-- Euclidean distance --")
    distances = pairwise_distances(data_for_calculations, metric="euclidean")
    print_distances(data, distances, "euclidean")
    print("-- Mahalanobis distance --")
    distances = pairwise_distances(data_for_calculations, metric="mahalanobis")
    print_distances(data, distances, "mahalanobis")

    # Exercise b
    print("-- Goodall distance --")
    data_for_calculations = data.drop(columns=data.columns.difference(CATEGORICAL_COLUMNS))
    data_for_calculations = data_for_calculations.drop(columns=['name'])
    propbs = calculate_probabilities(data_for_calculations)
    distance_calculator = lambda features_a, features_b: goodall_distance(propbs, features_a, features_b)
    distances = cdist(data_for_calculations, data_for_calculations, metric=distance_calculator)
    print_distances(data, distances, similarity=True)


if __name__ == '__main__':
    main()
