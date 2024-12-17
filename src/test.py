from src.netorch.coarsening.aco import ACOCoarsening

if __name__ == "__main__":
    import networkx as nx
    from necython import Graph as CGraph, aco_walk

    # Minimal Graph
    print("Creating minimal graph...")
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)  # Add a single edge with weight
    G.add_edge(1, 2, weight=2.0)  # Add another edge with weight

    # Instantiate ACOCoarsening with minimal parameters
    print("Initializing ACOCoarsening...")
    coarsening = ACOCoarsening(G, threshold=0.2, window_size=10, num_walks=2, walk_length=10, phe_power=1.0, evapo_rate=0.0, iterations=1)

    # Perform coarsening
    print("Starting merge process...")
    mapping = coarsening.merge(G)

    # Print results
    print("Mapping result:", mapping)
