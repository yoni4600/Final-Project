import datetime
import os
import random
from research_plan import ResearchPlan
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from datetime import datetime

BaseDir =""
class EvaluationPlan:
    
    def __init__(self, g, d, t1, t2, p, K):
        self.g = g
        self.d = d
        self.t1 = t1
        self.t2 = t2
        self.p = p
        self.K = K

    def EvaluationPlanAlg(self):
        print(f"Adding {self.p}% randomly edges to the graph")
        g_tag, e_tag = AddingEdges(self.g, self.p)
        RP = ResearchPlan(g_tag, self.d, self.t1, self.t2, self.p, self.K)
        print("Start Research plan algorithm ..")
        G_R, summed_matrices = RP.ResearchPlanAlg()
        GR_edges = G_R.edges

        print("Calculating the success rate ..")
        count = 0
        for u, v in e_tag:
            if (u, v) not in GR_edges:
                count += 1

        try:
            plot_edge_histograms(self.g.edges, summed_matrices, self.K)
        except Exception as e:
            print(f"An error occurred while plotting histograms: {str(e)}")
            import traceback
            traceback.print_exc()

        successRate = (count / len(e_tag)) * 100
        return successRate


def plot_edge_histograms(graph_edges, matrix, max_value, block_size=250, title="Edge Histogram"):
    """
    Plots histograms in blocks, where the x-axis represents edges in `graph_edges` and the y-axis is the corresponding
    values in the `matrix`. Saves the plots in a timestamped subdirectory inside `src/plots`.

    Args:
        graph_edges (list): List of edges to represent on the x-axis (e.g., `self.g.edges`).
        matrix (np.ndarray): The matrix containing the values for each edge.
        max_value (int): Maximum value in the matrix (e.g., `self.K`).
        block_size (int): Number of edges to include in each plot block.
        title (str): Title of the histogram plots.
    """
    edges = list(graph_edges)
    total_edges = len(edges)

    # Create a timestamped directory
    base_dir = os.path.join(BaseDir, 'plots')
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = os.path.join(base_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    #save config file in run directory
    Config.save_to_json(filename=os.path.join(output_dir,f"RunConfig.json"))
    
    # Extract values from the matrix based on graph_edges
    values_from_edges = [matrix[i, j] for i, j in graph_edges]

    # Count occurrences of each value (0 to k) in the filtered values
    unique, counts = np.unique(values_from_edges, return_counts=True)

    # Calculate percentages
    percentages = (counts / len(graph_edges)) * 100

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        percentages,
        labels=[f"Value {val}" for val in unique],
        autopct='%1.1f%%',  # Format percentages
        startangle=90,      # Rotate the chart for better appearance
        colors=plt.cm.tab10.colors[:len(unique)]  # Use different colors
    )
    plt.savefig(os.path.join(output_dir, "graph_edges_distribution.png"))
    # Calculate the number of blocks
    num_blocks = (total_edges + block_size - 1) // block_size  # Ceiling division

    # Iterate through blocks of edges
    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, total_edges)
        block_edges = edges[start_idx:end_idx]
        values = [matrix[u][v] for u, v in block_edges]

        # Create the histogram for the current block
        plt.figure(figsize=(14, 8))
        plt.bar(
            range(len(block_edges)),
            values,
            color="blue",
            edgecolor="black",
            alpha=0.75
        )
        plt.title(f"{title} (Block {block_idx + 1}/{num_blocks})")
        plt.xlabel("Edges (first_node, second_node)")
        plt.ylabel(f"Cell Values (0 to {max_value})")
        plt.xticks(rotation=90, fontsize=8)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()


        # Create the file path
        filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_block_{block_idx + 1}.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved plot to {filename}")


def AddingEdges(g, p):
    """
    Randomly adds a percentage of edges to the graph.

    Args:
        g (networkx.Graph): The input graph.
        p (float): The percentage of edges to add (0 <= p <= 100).

    Returns:
        Tuple[networkx.Graph, list]: The modified graph and a list of added edges.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage p must be between 0 and 100.")

    # Get the list of all possible edges in a fully connected graph
    nodes = list(g.nodes())
    edges = set(g.edges())
    possible_edges = set((u, v) for u in nodes for v in nodes if u != v) - edges

    # Calculate the number of edges to add
    num_possible_edges = len(possible_edges)
    num_to_add = int((p / 100) * len(edges))

    # Randomly sample edges to add
    edges_to_add = random.sample(possible_edges, min(num_to_add, num_possible_edges))

    # Add the edges to the graph
    for u, v in edges_to_add:
        g.add_edge(u, v, weight=1)

    return g, edges_to_add
