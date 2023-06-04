from itertools import count

import networkx as nx
from matplotlib import pyplot as plt

colors = ['', 'BLUE', 'GREEN', 'RED', 'YELLOW', 'ORANGE', 'PINK', 'BLACK', 'BROWN', 'WHITE', 'PURPLE', 'VIOLET']

edges = [(0, 1), (0, 4), (0, 5), (4, 5), (1, 4), (1, 3), (2, 3), (2, 4)]


def graphE(edges, n):
    adjList = [[] for _ in range(n)]
    for (src, dest) in edges:
        adjList[src].append(dest)
        adjList[dest].append(src)
    return adjList


def assign_color(assign_set):
    color = 1
    for c in assign_set:
        if color != c:
            break
        color = color + 1
    return color


color_map = list()


def color_graph(edges, n):
    result = {}
    graph = graphE(edges, n)
    for u in range(n):
        assigned = set([result.get(i) for i in graph[u] if i in result])
        result[u] = assign_color(assigned)
    for v in range(n):
        #if len(result) > v:
        color_map.append(colors[result[v]])
        print(f'Color assigned to vertex {v} is {colors[result[v]]}')


n = 6
color_graph(edges, n)

G = nx.Graph()
G.add_nodes_from([i for i in range(5)])
G.add_edges_from(edges)
nx.draw(G, node_color=color_map, node_size=600, with_labels=True)
plt.savefig("Graph.png", format="PNG")
