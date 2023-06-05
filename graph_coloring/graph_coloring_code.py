from itertools import count

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from docplex.mp.model import Model
from qiskit_aer import Aer
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp

from qiskit import Aer
from qiskit.algorithms import QAOA
#from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.utils import algorithm_globals
from qiskit.utils import QuantumInstance

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
        color_map.append(colors[result[v]])
        print(f'Color assigned to vertex {v} is {colors[result[v]]}')


n = 6
color_graph(edges, n)

G = nx.Graph()
G.add_nodes_from([i for i in range(5)])
G.add_edges_from(edges)
nx.draw(G, node_color=color_map, node_size=600, with_labels=True)
plt.savefig("classical_graph_coloring.png", format="PNG")
plt.clf()


def draw_graph(n, edges):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(1, n + 1)])
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True)
    return G


n = 5
edges = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
G = draw_graph(n, edges)
plt.savefig("basic_graph.png", format="PNG")
plt.clf()

num_color = 3
P = 4

num_nodes = G.number_of_nodes()
Q = np.eye(num_nodes * num_color)
for i in range(1, num_nodes + 1):
    l = num_color * i + 1
    for j in range(l - num_color, l):
        for k in range(l - num_color, l):
            if k == j:
                Q[j - 1][k - 1] = -P
            else:
                Q[j - 1][k - 1] = P

for i, j in G.edges:
    for k in range(1, num_color + 1):
        # (node - 1) * num_color + k
        m = (i - 1) * num_color + k
        o = (j - 1) * num_color + k
        Q[m - 1][o - 1] = P / 2
        Q[o - 1][m - 1] = P / 2

mdl = Model(name="Graph Coloring")
x = [mdl.binary_var('x%s' % i) for i in range(len(Q))]
objective = mdl.sum([x[i] * Q[i, j] * x[j] for i in range(len(Q)) for i in range(len(Q))])
mdl.minimize(objective)
qp = from_docplex_mp(mdl)

print(qp.prettyprint())

seed = 1234
algorithm_globals.random_seed = seed
quantum_instance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=10000)
qaoa = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=3, quantum_instance=quantum_instance))
qaoa_result = qaoa.solve(qp)


def assign_color(result, num_color):
    colors = ['BLUE', 'GREEN', 'RED', 'YELLOW', 'ORANGE', 'PINK', 'BLACK', 'BROWN', 'WHITE', 'PURPLE', 'VIOLET']
    color_map = list()
    for pos, val in enumerate(result.x):
        if val == 1:
            node = pos // num_color + 1
            color = pos % num_color
            color_map.append(colors[color])
            print("for node %s with binary variable %s the assigned color is %s" % (node, pos, colors[color]))
    return color_map


qaoa_color_map = assign_color(qaoa_result, 3)


def draw_color_graph(n, edges, color_map):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(1, n + 1)])
    G.add_edges_from(edges)
    nx.draw(G, node_color=color_map, node_size=600, with_labels=True)


draw_color_graph(n, edges, qaoa_color_map)
plt.savefig("quantum_graph_coloring.png", format="PNG")
plt.clf()
