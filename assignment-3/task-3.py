import networkx as nx
import pandas as pd
import numpy as np

G = nx.Graph()

# Loading of the dataframe
filename = "data/jester-800-10.csv"
jokes_columns = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]
jokes_names = [f"joke_{joke_column}" for joke_column in jokes_columns]
jokes_df = pd.read_csv(filename)

# Construction of the bipartite graph.
for i, user in jokes_df.iterrows():
    G.add_node(user["id"])
    for name in jokes_names:
        if user[name] == 1:
            G.add_edge(user["id"], name)

# Calculation of the SimRank similarity matrix
sim = nx.simrank_similarity(G, max_iterations=1)

# Selection of the user to test.
user_id = 16210

# Calculation of the k nearest neighbors.
n_neighbors = 5

neighbours = sim[user_id]
sorted_neighbors = {k: v for k, v in sorted(neighbours.items(), key=lambda item: item[1], reverse=True)}

i = 0
k_sorted_neighbors = {}
for k in sorted_neighbors:
    if i <= n_neighbors:
        if i > 0:
            k_sorted_neighbors[k] = sorted_neighbors[k]
        i += 1
    else:
        break

# Compute the averages for k neighbours.

k_averages = {}

for k in k_sorted_neighbors:
    neighbor_row = jokes_df.loc[jokes_df["id"] == k]
    values = []
    for joke_name in jokes_names:
        values.append(neighbor_row[joke_name].values[0])
    k_averages[k] = np.mean(values)

# Calculation of the ratings of the unrated jokes, whose value is 0.

user_row = jokes_df.loc[jokes_df["id"] == user_id]
predictions = {}

for joke_name in jokes_names:
    # This means that we have to compute prediction for joke
    if user_row[joke_name].values[0] == 0:
        numerator = 0  # of the rating formula.
        denominator = 0  # of the rating formula.
        for k in k_sorted_neighbors:
            neighbour_dist = k_sorted_neighbors[k]  # Distance measure
            neighbour_avg = k_averages[k]

            neighbour_col = jokes_df.loc[jokes_df["id"] == k]

            numerator += neighbour_dist * (neighbour_col[joke_name].values[0] - neighbour_avg)
            denominator += neighbour_dist

        predictions[joke_name] = (numerator / denominator) + jokes_df[joke_name].mean()

# Finally, predictions are sorted
predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

# Print most relevant predictions for user.
print(f"Most relevant jokes for user {user_id}")

for k in predictions:
    print(f"{k} -> {predictions[k]}")
