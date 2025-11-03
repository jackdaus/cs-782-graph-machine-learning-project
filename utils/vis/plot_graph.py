import networkx
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def plot_networkx(data: Data):
    graph = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(5,5))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=True, width=0.1)
    plt.show()