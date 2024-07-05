import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

edges_df = pd.read_excel('./data/branch_data.xlsx')
# print(edges_df)
G = nx.Graph()

for index, row in edges_df.iterrows():
    impedance = complex(row['r'], row['x'])
    weight = abs(impedance)
    G.add_edge(row['fbus'], row['tbus'], weight=weight)
    

pos = nx.kamada_kawai_layout(G, weight='weight')

nx.draw(G, with_labels=True)
plt.savefig("network_graph.png")
plt.show()
