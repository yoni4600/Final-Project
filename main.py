import networkx as nx
from netorch.dataset import load_edgelist, load_labels
from src.evaluation_plan import EvaluationPlan


def main():
    DATASET = 'cora'
    DATASET_DIR = 'datasets'
    EDGES_TXT = '{}/{}_edgelist.txt'.format(DATASET_DIR, DATASET)
    LABELS_TXT = '{}/{}_labels.txt'.format(DATASET_DIR, DATASET)

    # loads the graph
    g, labels = load_edgelist(EDGES_TXT), load_labels(LABELS_TXT)
    g = nx.convert_node_labels_to_integers(g)

    EP = EvaluationPlan(g, d=128, t1=0.2, t2=50, p=30, K=3)
    successRate = EP.EvaluationPlanAlg()
    print(successRate)


if __name__ == "__main__":
    main()
