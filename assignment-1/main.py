# Imports
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', 199)  # or 199

DATA_FILE = "assignment-1/data/nba2013_data.csv"
CATEGORICAL_COLUMNS = ['player', 'pos', 'bref_team_id', 'season', 'season_end']
PLOT_FOLDER = "assignment-1/plots/"


def preprocessing(data):
    preprocessed_data = data.drop(columns=CATEGORICAL_COLUMNS)

    # At a fist option we eliminate columns with NaN.
    preprocessed_data = preprocessed_data.apply(pd.to_numeric, errors='coerce')
    preprocessed_data = preprocessed_data.dropna()

    # columns_with_nan = preprocessed_data.columns[preprocessed_data.isna().any()].tolist()
    return preprocessed_data


def plot_silhouette_scores(n_clusters, data, labels, silhouette_avg, silhouette_values, file_name=None):
    fig, ax = plt.subplots()
    # Set limits for printing the silhouette.
    ax.set_xlim(-0.2, 1)
    ylim = len(data) + 10 * (n_clusters + 1)
    ax.set_ylim(0, ylim)

    y_lower = 10
    for i in range(n_clusters):
        # Compute the ith silhouette value and sort for each value.
        ith_cluster_silhouette_values = silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    # Labels for axes
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.text(silhouette_avg - 0.03, ylim + 5, str(np.round(silhouette_avg, 2)))

    # Ticks for x and y axes
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.2, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for K - Means clustering "
                  "with k = %d" % n_clusters), fontweight='bold')

    if file_name is not None:
        plt.savefig(f'{PLOT_FOLDER}{file_name}')


def exercise_1():
    data = pd.read_csv(DATA_FILE)
    data = preprocessing(data)

    range_n_clusters = [2, 5, 10]

    for n_clusters in range_n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=10)
        labels = km.fit_predict(data)

        # Silhouette average for all samples.
        silhouette_avg = silhouette_score(data, labels)
        print("K = ", n_clusters, " silhouette average -> ", silhouette_avg)

        # Silhouette score for each sample.
        silhouette_values = silhouette_samples(data, labels)

        plot_silhouette_scores(n_clusters, data, labels, silhouette_avg, silhouette_values,
                               f'silhouette-plot-k-{n_clusters}.png')

    plt.show()


if __name__ == '__main__':
    exercise_1()
