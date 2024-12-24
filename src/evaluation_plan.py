import random
from src.research_plan import ResearchPlan


class EvaluationPlan:

    def __init__(self, g, d, t1, t2, p, K):
        self.g = g
        self.d = d
        self.t1 = t1
        self.t2 = t2
        self.p = p
        self.K = K

    def EvaluationPlanAlg(self):
        g_tag, e_tag = AddingEdges(self.g, self.p)
        RP = ResearchPlan(g_tag, self.d, self.t1, self.t2, self.p, self.K)
        G_R = RP.ResearchPlanAlg()
        GR_edges = G_R.edges

        count = 0
        for edge in e_tag:
            if edge not in GR_edges:
                count += 1

        successRate = count / len(e_tag) * 100
        return successRate


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
