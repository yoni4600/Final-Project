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
        print(f"Adding {self.p}% randomly edges to the graph\n")
        g_tag, e_tag = AddingEdges(self.g, self.p)
        RP = ResearchPlan(g_tag, self.d, self.t1, self.t2, self.p, self.K)
        print("Start Research plan algorithm ..\n")
        G_R, summed_matrices = RP.ResearchPlanAlg()
        GR_edges = G_R.edges

        print("Calculating the success rate ..\n")
        count = 0
        for u, v in e_tag:
            if (u, v) not in GR_edges:
                count += 1

        try:
            plot_edge_histograms(self.g.edges, summed_matrices, self.K, g_tag.edges, e_tag , GR_edges, count)
        except Exception as e:
            print(f"An error occurred while plotting histograms: {str(e)}\n")
            import traceback
            traceback.print_exc()

        successRate = (count / len(e_tag)) * 100
        return successRate


def plot_edge_histograms(graph_edges, matrix, max_value, manipulated_graph_edges ,fake_edges , refined_graph_edges, fake_edges_removed, block_size=250, title="Edge Histogram"):
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

    # Plot the bar chart
    plt.figure(figsize=(12, 8))  # Adjust size to fit many categories
    bars = plt.barh(unique, percentages, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
    for bar, category in zip(bars, unique):
        custom_text = f" Times restored = {category}"  # Add a prefix or other information
        plt.text(-1, bar.get_y() + bar.get_height() / 2, custom_text, va='center', ha='right')  # Place the custom text
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{percentage:.1f}%", va='center')
    
    # Remove x and y axes
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # Add a title
    plt.title(f'Edges Distribution throughout K(={Config.K}) iterations, Threshold t1 = {Config.TRESHOLD1}, Threshold t2 ={Config.TRESHOLD2}')
    # Add conclusion text under the bars
    total_edges_removed = len(manipulated_graph_edges)-len(refined_graph_edges)
    conclusion_text1 = f"Total amount of edges removed from the manipulated Graph = {total_edges_removed}, equals to {(total_edges_removed / len(manipulated_graph_edges)) * 100}%"
    count = 0
    for u,v in graph_edges:
        if(u,v) not in fake_edges and (u,v) in refined_graph_edges:
            count += 1
    real_edges_removed = len(graph_edges) - count
    conclusion_text2 =f"Total amout of edges removed from the REAL Graph = {real_edges_removed}, equals to {(real_edges_removed / len(graph_edges)) * 100}%"
    successRate = (fake_edges_removed / len(fake_edges)) * 100
    conclusion_text3 = f"Total amount of 'fake' edges removed = {fake_edges_removed}, thus the success rate of this run is {successRate:.3f}%"
    conclusion_text = f"{conclusion_text1}\n{conclusion_text2}\n{conclusion_text3}"
    # Use figtext to place text outside the axes, under the bars
    plt.figtext(0.5, -0.05, conclusion_text, ha='center', va='top', fontsize=12, color='black', wrap=True)
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9, top=0.8, wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_edges_distribution_BarChart.png"))
    
    # # Plot the pie chart
    # plt.figure(figsize=(8, 8))
    # plt.pie(
    #     percentages,
    #     labels=[f"Value {val}" for val in unique],
    #     autopct='%1.1f%%',  # Format percentages
    #     startangle=90,      # Rotate the chart for better appearance
    #     colors=plt.cm.tab10.colors[:len(unique)]  # Use different colors
    # )
    # plt.savefig(os.path.join(output_dir, "graph_edges_distribution_PieChart.png"))
    # Calculate the number of blocks
    
    
    # num_blocks = (total_edges + block_size - 1) // block_size  # Ceiling division

    # # Iterate through blocks of edges
    # for block_idx in range(num_blocks):
    #     start_idx = block_idx * block_size
    #     end_idx = min(start_idx + block_size, total_edges)
    #     block_edges = edges[start_idx:end_idx]
    #     values = [matrix[u][v] for u, v in block_edges]

    #     # Create the histogram for the current block
    #     plt.figure(figsize=(14, 8))
    #     plt.bar(
    #         range(len(block_edges)),
    #         values,
    #         color="blue",
    #         edgecolor="black",
    #         alpha=0.75
    #     )
    #     plt.title(f"{title} (Block {block_idx + 1}/{num_blocks})")
    #     plt.xlabel("Edges (first_node, second_node)")
    #     plt.ylabel(f"Cell Values (0 to {max_value})")
    #     plt.xticks(rotation=90, fontsize=8)
    #     plt.grid(axis="y", linestyle="--", alpha=0.7)
    #     plt.tight_layout()


    #     # Create the file path
    #     filename = os.path.join(output_dir, f"{title.replace(' ', '_')}_block_{block_idx + 1}.png")
    #     plt.savefig(filename)
    #     plt.close()


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
