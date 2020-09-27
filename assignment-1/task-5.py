import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt


def plot():
    data = pd.read_csv("data/mat.csv", header=None)
    root = np.sqrt(1 / 2)
    data_2 = np.array([[root, root],
                       [root, 2 * root],
                       [4 * root, root],
                       [4 * root, 2 * root]])
    df_2 = pd.DataFrame({'Column1': data_2[:, 0], 'Column2': data_2[:, 1]})
    data.plot(x=0, y=1, kind='scatter')
    df_2.plot(x='Column1', y='Column2', kind='scatter')
    # plt.plot(data.iloc[0].to_numpy(), data.iloc[1].to_numpy())
    plt.show()
    print(data)


def main():
    data = pd.read_csv("data/mat.csv", header=None)
    root = np.sqrt(1 / 2)
    data_2 = np.array([[root, root],
                       [root, 2 * root],
                       [4 * root, root],
                       [4 * root, 2 * root]])
    data = pd.DataFrame({'Column1': data_2[:, 0], 'Column2': data_2[:, 1]})
    print("original data")
    print(data)

    # Standarize data
    scaler = StandardScaler()
    scaler.fit(data)

    print("standarized data")
    X = scaler.transform(data)
    print(X)
    print(scaler.var_)
    # calculate covariance matrix
    print("covar matrix")
    cov = (X.T @ X) / (X.shape[0] - 1)
    print(cov)
    # eigen
    eig_values, eig_vectors = np.linalg.eig(cov)
    print(eig_values)
    print(eig_vectors)
    distances = pairwise_distances(X, metric="euclidean")
    print("yh")
    print(distances)

    eig_scores = np.dot(X, [eig_vectors[0][0], eig_vectors[1][0]])
    print(eig_scores)
    distances = pairwise_distances(eig_scores.reshape(-1, 1), metric="euclidean")
    print(distances)

    eig_scores = np.dot(X, eig_vectors)
    print(eig_scores)
    distances = pairwise_distances(eig_scores, metric="euclidean")
    print(distances)


if __name__ == '__main__':
    main()
