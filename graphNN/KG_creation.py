import networkx as nx
import matplotlib.pyplot as plt

# Sample data: list of tuples (entity1, entity2, relation)
classified_data = [
    ("Entity1", "Entity2", "related"),
    ("Entity3", "Entity4", "not_related"),
    # Add more classified NER and their relations here...
]

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for entity1, entity2, relation in classified_data:
    G.add_node(entity1)
    G.add_node(entity2)
    G.add_edge(entity1, entity2, label=relation)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

# Show the plot
plt.show()

# Save the graph as a PNG image
