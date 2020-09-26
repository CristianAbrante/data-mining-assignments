# Imports
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

DATA_FILE = "assignment-1/data/nba2013_data.csv"
CATEGORICAL_COLUMNS = ['player', 'pos', 'bref_team_id', 'season', 'season_end']
PLOT_FOLDER = "assignment-1/plots/"
TASK_1_PLOTS = f'{PLOT_FOLDER}/task-1/'
TASK_2_PLOTS = f'{PLOT_FOLDER}/task-2/'


def preprocessing(data):
    preprocessed_data = data.drop(columns=CATEGORICAL_COLUMNS)

    # At a fist option we eliminate columns with NaN.
    preprocessed_data = preprocessed_data.apply(pd.to_numeric, errors='coerce')
    preprocessed_data = preprocessed_data.dropna()

    # columns_with_nan = preprocessed_data.columns[preprocessed_data.isna().any()].tolist()
    return preprocessed_data


def log_metric(range_n_clusters, metric, metric_name):
    print(f'--- {metric_name} ---')
    for i in range(len(range_n_clusters)):
        print_formatted_metric(metric[i], metric_name, range_n_clusters[i])


def print_formatted_metric(metric_value, metric_name, n_clusters):
    print(f'K = {n_clusters} {metric_name} -> {np.round(metric_value, 4)}')


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
        plt.savefig(f'{TASK_1_PLOTS}{file_name}')


def exercise_1(log_results=True):
    data = pd.read_csv(DATA_FILE)
    data = preprocessing(data)

    range_n_clusters = [2, 5, 10]

    silhouette_avgs = []
    ch_metrics = np.array([])
    db_metrics = np.array([])

    for n_clusters in range_n_clusters:
        km = KMeans(n_clusters=n_clusters, random_state=10)
        labels = km.fit_predict(data)

        # Silhouette analysis
        silhouette_avg = silhouette_score(data, labels)
        silhouette_avgs = np.append(silhouette_avgs, silhouette_avg)

        silhouette_values = silhouette_samples(data, labels)
        if log_results:
            plot_silhouette_scores(n_clusters, data, labels, silhouette_avg, silhouette_values,
                                   f'silhouette-plot-k-{n_clusters}.png')

        # Calinski Harabasz analysis
        ch_metrics = np.append(ch_metrics, calinski_harabasz_score(data, labels))

        # davies bouldin analysis
        db_metrics = np.append(db_metrics, davies_bouldin_score(data, labels))

    if log_results:
        log_metric(range_n_clusters, silhouette_avgs, "Silhouette score")
        log_metric(range_n_clusters, ch_metrics, "calinski-harabsz score")
        log_metric(range_n_clusters, db_metrics, "davies bouldin score")
        plt.show()


def plot_dendrogram(model, linkage_metric, file_name=None, **kwargs):
    fig, ax = plt.subplots()
    # Create linkage matrix and then plot the dendrogram
    plt.title(f"Hierarchical clustering Dendogram - metric = {linkage_metric}")
    dendrogram(model, **kwargs)
    ax.set_xlabel("Number of points in node (or index of point if no parenthesis).")

    if file_name is not None:
        plt.savefig(f'{TASK_2_PLOTS}{file_name}')


def scatterplot_silhouette_scores(silhouette_scores, range_of_clusters, metric, file_name=None):
    fig, ax = plt.subplots()

    plt.title(f"Average Silhouette score, metric = {metric}")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")

    plt.plot(range_of_clusters, silhouette_scores, '-ok')

    if file_name is not None:
        plt.savefig(f'{TASK_2_PLOTS}{file_name}')


def exercise_2():
    data = pd.read_csv(DATA_FILE)
    data = preprocessing(data)
    linkage_metrics = ['single', 'complete', 'average', 'centroid']

    for linkage_metric in linkage_metrics:
        X = linkage(data.to_numpy(), linkage_metric)

        # variable for truncate visualization
        p = 4
        plot_dendrogram(X, linkage_metric, f"dendogram-{linkage_metric}-p{p}", truncate_mode='level', p=p)

        range_n_clusters = range(2, 11)
        log_results = True

        silhouette_avgs = np.array([])

        for n_clusters in range_n_clusters:
            labels = fcluster(X, n_clusters, criterion="maxclust")
            # Silhouette analysis
            silhouette_avgs = np.append(silhouette_avgs, silhouette_score(data, labels))

        if log_results:
            log_metric(range_n_clusters, silhouette_avgs, "Silhouette score")
            scatterplot_silhouette_scores(silhouette_avgs, range_n_clusters, linkage_metric,
                                          f"silhouette-{linkage_metric}")

    plt.show()


if __name__ == '__main__':
    exercise_2()
