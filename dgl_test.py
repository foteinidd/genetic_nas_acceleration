from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, MAX_CONNECTIONS
from matplotlib import pyplot as plt
import networkx as nx


is_valid_architecture = False
while not is_valid_architecture:
    architecture = randomly_sample_architecture()
    # check if connection number is ok for nasbench-101
    if sum(architecture.connections) <= MAX_CONNECTIONS and architecture.valid_architecture:
        is_valid_architecture = True
print(architecture)
d = create_nord_architecture(architecture)
print(d)
G = d.to_networkx()
nx.draw(G)
plt.show()
