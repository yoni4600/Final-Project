import csv
import datetime
import os
import random
from research_plan import ResearchPlan
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from datetime import datetime

BaseDir =""


class MainResearchPlan:
    
    def __init__(self, g, d, t1, t2, p, K):
        self.g = g
        self.d = d
        self.t1 = t1
        self.t2 = t2
        self.p = p
        self.K = K

    def MainResearchPlanAlg(self):
        RP = ResearchPlan(self.g, self.d, self.t1, self.t2, self.p, self.K)
        print("Start Research plan algorithm ..\n")
        G_R, summed_matrices = RP.ResearchPlanAlg()
        GR_edges = G_R.edges

        try:
            plot_edge_histograms(self.g.edges, summed_matrices, self.K, GR_edges, )
            push_git_changes()
        except Exception as e:
            print(f"An error occurred while plotting histograms: {str(e)}\n")
            import traceback
            traceback.print_exc()



def plot_edge_histograms(graph_edges, matrix, max_value, refined_graph_edges, block_size=250, title="Edge Histogram"):
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
    # Create a timestamped directory
    base_dir = os.path.join(BaseDir, 'plots')
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_dir = os.path.join(base_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)
    # Save config file in run directory
    Config.save_to_json(filename=os.path.join(output_dir, f"RunConfig.json"))

    # Extract values from the matrix based on graph_edges
    values_from_edges = [matrix[i, j] for i, j in graph_edges]

    # Extract edges with value 0
    edges_with_max_value = [(i, j) for i, j in graph_edges if matrix[i, j] == max_value]

    # Save edges with value 0 to a CSV file
    csv_file_max = os.path.join(output_dir, "edges_with_max_value.csv")
    with open(csv_file_max, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Node1", "Node2"])
        writer.writerows(edges_with_max_value)

    print(f"Edges with max value saved to: {csv_file_max}")
    
    edges_with_value_0 = [(i, j) for i, j in graph_edges if matrix[i, j] == 0]

    # Save edges with value 0 to a CSV file
    csv_file_0 = os.path.join(output_dir, "edges_with_value_0.csv")
    with open(csv_file_0, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Node1", "Node2"])
        writer.writerows(edges_with_value_0)

    print(f"Edges with max value saved to: {csv_file_0}")

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
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{percentage:.3f}%", va='center')

    # Remove x and y axes
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # Add a title
    plt.title(f'Edges Distribution throughout K(={Config.K}) iterations, Threshold t1 = {Config.TRESHOLD1}, Threshold t2 ={Config.TRESHOLD2}')
    # Add conclusion text under the bars
    edges_removed = 0
    for u, v in graph_edges:
        if (u,v) not in refined_graph_edges:
            edges_removed += 1
    conclusion_text = f"Total amount of edges removed from the Graph = {edges_removed}, equals to {(edges_removed / len(graph_edges)) * 100}%"
    conclusion_fileName = "RunConclusions.txt"


    # Save conclusion text to a file
    with open(os.path.join(output_dir, conclusion_fileName), 'w') as file:
        file.write(conclusion_text)
    
    print(f"Conclusions saved to: {os.path.join(output_dir, conclusion_fileName)}\n\n")
    
    print(conclusion_text)
    # Use figtext to place text outside the axes, under the bars
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.9, top=0.8, wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_edges_distribution_BarChart.png"))

def push_git_changes():
        os.system('git add .')
        os.system('git commit -m "Updated EvaluationPlan with git push script"')
        os.system('git push')