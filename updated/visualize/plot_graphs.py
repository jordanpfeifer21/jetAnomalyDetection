import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def visualize_graph(data: Data, title: str, k: int = 16):
    """
    Visualize a kNN graph from particle flow candidates.
    """
    edge_index = data.edge_index.cpu().numpy()
    pos = {i: [data.x[i, 1].item(), data.x[i, 2].item()] for i in range(data.num_nodes)}

    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    for start, end in edge_index.T:
        G.add_edge(int(start), int(end))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='k')
    nx.draw_networkx_edges(G, pos, edge_color='r', width=1)
    plt.title(f'Graph Visualization: {title} (k={k})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.clf()