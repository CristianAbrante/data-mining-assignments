#!/usr/bin/env python

import networkx as nx
import numpy as np


def construct_subgraph(original_graph, size):
    new_graph = nx.Graph()
    visited = [False for _ in range(len(original_graph.nodes))]

    initial_node = 1
    stack = [initial_node]

    while len(stack) != 0 and len(new_graph.nodes) < size:
        current_node = stack.pop()
        if not visited[current_node - 1]:
            visited[current_node - 1] = True

            neighbours = [(node, original_graph[current_node][node]["weight"]) for node in original_graph[current_node]]
            n_type = [('index', int), ('weight', float)]

            sorted_neighbours = np.array(neighbours, dtype=n_type)
            sorted_neighbours = np.sort(sorted_neighbours, order="weight")

            for neighbour in sorted_neighbours:
                new_graph.add_edge(int(current_node), int(neighbour[0]))
                stack.append(neighbour[0])

    return new_graph


G = nx.Graph()

# Define a filename.
filename = "data/dolphins.txt"

# Open the file as f.
# The function readlines() reads the file.
with open(filename) as f:
    content = f.readlines()

# Add nodes to graph
n_of_nodes = int(content[0])
for i in range(n_of_nodes):
    G.add_node(i + 1)

for edge_str in content:
    pair = edge_str.split()
    if len(pair) > 1:
        G.add_edge(int(pair[0]), int(pair[1]), weight=1.0)

print(G.nodes)
print(G.edges)

G2 = construct_subgraph(G, 10)
print(G2.nodes)
print(G2.edges)
