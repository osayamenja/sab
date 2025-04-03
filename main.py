import networkx as nx
import matplotlib.pyplot as plt
if __name__ == '__main__':
    G = nx.complete_graph(5)
    nx.draw(G)
    plt.show()
