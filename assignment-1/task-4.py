import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist, euclidean

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


def overlap_measure(features_a, features_b):
    overlap = 0
    k = len(features_a)
    for i in range(k):
        if features_a[i] == features_b[i]:
            overlap += 1
    return overlap / k


def combined_similarity(categorical_positions, numerical_positions, lambda_val, features_a, features_b):
    a_categorical = [features_a[i] for i in categorical_positions]
    a_numerical = [features_a[i] for i in numerical_positions]
    b_categorical = [features_b[i] for i in categorical_positions]
    b_numerical = [features_b[i] for i in numerical_positions]

    euclidean_val = euclidean(a_numerical, b_numerical)
    overlap_val = overlap_measure(a_categorical, b_categorical)

    return lambda_val * (1.0 / (1.0 + euclidean_val)) + (1 - lambda_val) * overlap_val


def combined_distance(categorical_positions, numerical_positions, lambda_val, features_a, features_b):
    a_categorical = [features_a[i] for i in categorical_positions]
    a_numerical = [features_a[i] for i in numerical_positions]
    b_categorical = [features_b[i] for i in categorical_positions]
    b_numerical = [features_b[i] for i in numerical_positions]

    euclidean_val = euclidean(a_numerical, b_numerical)
    overlap_val = overlap_measure(a_categorical, b_categorical)

    return lambda_val * euclidean_val + (1 - lambda_val) * (1 - overlap_val)


def main():
    data = pd.read_csv(DATA_FILE)

    ## Exercise a
    data_for_calculations = data.drop(columns=CATEGORICAL_COLUMNS)
    scaler = MinMaxScaler()
    scaler.fit(data_for_calculations)
    data_for_calculations = scaler.transform(data_for_calculations)

    print("-- Euclidean distance --")
    distances = pairwise_distances(data_for_calculations, metric="euclidean")
    print_distances(data, distances)
    print("-- Mahalanobis distance --")
    distances = pairwise_distances(data_for_calculations, metric="mahalanobis")
    print_distances(data, distances)

    # Exercise b
    print("-- Goodall distance --")
    data_for_calculations = data.drop(columns=data.columns.difference(CATEGORICAL_COLUMNS))
    data_for_calculations = data_for_calculations.drop(columns=['name'])
    propbs = calculate_probabilities(data_for_calculations)
    distance_calculator = lambda features_a, features_b: goodall_distance(propbs, features_a, features_b)
    distances = cdist(data_for_calculations, data_for_calculations, metric=distance_calculator)
    print_distances(data, distances, similarity=True)

    # Exercise c
    print("-- Combined measure --")
    scaler = MinMaxScaler()
    data_for_calculations = data

    numerical_columns = data.columns.difference(CATEGORICAL_COLUMNS)

    # Min max scaling numerical values
    for col in numerical_columns:
        data_for_calculations[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(data_for_calculations[col])),
                                                  columns=[col])

    lambda_value = (data.shape[1] - len(CATEGORICAL_COLUMNS)) / data.shape[1]
    categorical_positions = [data.columns.get_loc(c) for c in CATEGORICAL_COLUMNS if c in data]
    numerical_positions = [data.columns.get_loc(c) for c in numerical_columns if c in data]

    calculator = lambda features_a, features_b: combined_similarity(categorical_positions, numerical_positions,
                                                                    lambda_value, features_a,
                                                                    features_b)

    distances = cdist(data_for_calculations, data_for_calculations, metric=calculator)
    print(distances)
    print_distances(data, distances, similarity=True)

    # Exercise d
    print("-- Combined measure (distance) --")
    scaler = MinMaxScaler()
    data_for_calculations = data

    numerical_columns = data.columns.difference(CATEGORICAL_COLUMNS)

    # Min max scaling numerical values
    for col in numerical_columns:
        data_for_calculations[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(data_for_calculations[col])),
                                                  columns=[col])

    lambda_value = (data.shape[1] - len(CATEGORICAL_COLUMNS)) / data.shape[1]
    categorical_positions = [data.columns.get_loc(c) for c in CATEGORICAL_COLUMNS if c in data]
    numerical_positions = [data.columns.get_loc(c) for c in numerical_columns if c in data]

    calculator = lambda features_a, features_b: combined_distance(categorical_positions, numerical_positions,
                                                                  lambda_value, features_a,
                                                                  features_b)

    distances = cdist(data_for_calculations, data_for_calculations, metric=calculator)
    print(distances)
    print_distances(data, distances)


if __name__ == '__main__':
    main()
