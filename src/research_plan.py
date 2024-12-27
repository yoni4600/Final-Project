# coding:utf-8
import random
import numpy as np
from netorch.coarsening.aco import ACOCoarsening
from netorch.models.walkbased import Node2Vec
from netorch.models.hierarchical import MLNE
from scipy.spatial.distance import cosine
from scipy.special import expit  # Sigmoid function
from config import Config


class ResearchPlan:

    def __init__(self, g, d, t1, t2, p, K):
        self.g = g
        self.d = d
        self.t1 = t1
        self.t2 = t2
        self.p = p
        self.K = K

    def ResearchPlanAlg(self):
        similarity_matrix_list = []

        for i in range(self.K):
            print(f"\nIteration number {i+1} of total {self.K} iterations:")
            original_graph = self.g.copy()
            print(f"\tRandomly remove {self.p}% of the graph's edges")
            g_tag = RemoveEdges(original_graph, self.p)

            # loads the model with the embedding method and coarsening method
            model = MLNE(
                graph=g_tag,
                dimension=self.d,
                Model=lambda graph, dimension: Node2Vec(graph, dimension=dimension, batch_size=Config.NODE2VEC_BATCH_SIZE, iterations=Config.NODE2VEC_ITERATIONS, p=Config.NODE2VEC_P, q=Config.NODE2VEC_Q),
                Coarsening=lambda graph: ACOCoarsening(graph, phe_power=Config.ALPHA, iterations=Config.ACO_COARSENING_ITERATIONS),
            )
            # start the process and receive the embedding matrix
            embedding_matrix = model.train().get_embeddings()
            print("\tCalculating the similarity matrix based on threshold 1 ..")
            similarity_matrix = CalculateCosineSimilarity(embedding_matrix, self.t1)
            similarity_matrix_list.append(similarity_matrix)

        print("Finished the K iterations")
        print("Calculating the statistical matrix ..")
        M_Stat, summed_matrices = CalculateStatistics(similarity_matrix_list, self.K)
        print("Refining the graph based on threshold 2 ..")
        G_R = RefineGraph(self.g.copy(), M_Stat, self.t2)
        return G_R, summed_matrices

# result = evaluate({lookup.index_to_label(index):embedding[index] for index in range(g.number_of_nodes())}, labels, clf_ratio=0.5)
# print(result)


def RemoveEdges(g, p):
    """
    Removes a random percentage of edges from the graph g.

    Args:
        g (networkx.Graph): The input graph.
        p (float): The percentage of edges to remove (0 <= p <= 100).

    Returns:
        networkx.Graph: The modified graph with edges removed.
    """
    if not (0 <= p <= 100):
        raise ValueError("Percentage p must be between 0 and 100.")

    # Calculate the number of edges to remove
    num_edges = g.number_of_edges()
    num_to_remove = int((p / 100) * num_edges)

    # Get all edges and randomly select edges to remove
    edges = list(g.edges())
    edges_to_remove = random.sample(edges, num_to_remove)

    # Remove the selected edges
    g.remove_edges_from(edges_to_remove)

    return g


def CalculateCosineSimilarity(embedding_matrix, t1):
    """
    Calculate the cosine similarity matrix with sigmoid normalization and thresholding.

    Args:
        embedding_matrix (numpy.ndarray): A matrix where each row is a vector embedding.
        t1 (float): Threshold for cosine similarity.

    Returns:
        numpy.ndarray: A binary similarity matrix.
    """
    num_vectors = embedding_matrix.shape[0]
    similarity_matrix = np.zeros((num_vectors, num_vectors), dtype=int)

    for m in range(num_vectors):
        for n in range(m, num_vectors):
            # Compute cosine similarity
            v_m = embedding_matrix[m]
            v_n = embedding_matrix[n]
            dot_product = np.dot(v_m, v_n)
            magnitude_m = np.linalg.norm(v_m)
            magnitude_n = np.linalg.norm(v_n)
            res = dot_product / (magnitude_m * magnitude_n)

            # Normalize the results from range [-1,1] to [0,1]
            # res = (res + 1)/2

            # Normalize with sigmoid
            # res = expit(res)

            # Apply threshold
            if res < t1:
                similarity_matrix[m, n] = False
            else:
                similarity_matrix[m, n] = True

            # Symmetry: Copy value for the lower triangular part
            similarity_matrix[n, m] = similarity_matrix[m, n]

    return similarity_matrix


def CalculateStatistics(similarity_matrix_list, K):
    """
    Calculate the statistical matrix from a list of boolean similarity matrices.

    Args:
        similarity_matrix_list (list of numpy.ndarray): A list of boolean similarity matrices.
        int K number of matrices in the similarity_matrix_list

    Returns:
        numpy.ndarray: The calculated statistical matrix with percentage values.
    """
    if not similarity_matrix_list:
        raise ValueError("The similarity matrix list is empty.")

    # Ensure all matrices in the list have the same dimensions
    shape = similarity_matrix_list[0].shape
    for matrix in similarity_matrix_list:
        if matrix.shape != shape:
            raise ValueError("All matrices in the list must have the same dimensions.")

    summed_matrix = np.zeros(shape, dtype=int)

    # Iterate through each matrix and sum them as True=1 False=0
    for matrix in similarity_matrix_list:
        summed_matrix += matrix

    # Calculate the statistical matrix
    M_stat = (summed_matrix / K) * 100

    return M_stat, summed_matrix


def RefineGraph(g, M_stat, t2):
    """
    Refine the graph by removing edges based on the statistical matrix and threshold.

    Args:
        g (networkx.Graph): The original graph.
        M_stat (numpy.ndarray): The statistical matrix with percentage values.
        t2 (float): The threshold for removing edges.

    Returns:
        networkx.Graph: The refined graph with edges removed based on the threshold.
    """

    # Iterate over each pair (m, n) in the statistical matrix
    num_nodes = M_stat.shape[0]
    for m in range(num_nodes):
        for n in range(m + 1, num_nodes):  # Only consider upper triangular part to avoid duplicates
            if M_stat[m, n] < t2:
                if g.has_edge(m, n):
                    g.remove_edge(m, n)
                elif g.has_edge(n, m):
                    g.remove_edge(n, m)

    return g
