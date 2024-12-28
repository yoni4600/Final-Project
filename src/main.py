import networkx as nx
from netorch.dataset import load_edgelist, load_labels
from evaluation_plan import EvaluationPlan
from config import Config


def main():
    DATASET = 'cora'
    DATASET_DIR = 'datasets'
    EDGES_TXT = 'src/{}/{}_edgelist.txt'.format(DATASET_DIR, DATASET)
    LABELS_TXT = 'src/{}/{}_labels.txt'.format(DATASET_DIR, DATASET)

    # loads the graph
    print(f"Loads the selected Data set: {DATASET}")
    g, labels = load_edgelist(EDGES_TXT), load_labels(LABELS_TXT)
    g = nx.convert_node_labels_to_integers(g)

    EP = EvaluationPlan(g, d=Config.DIMENSION, t1=Config.TRESHOLD1, t2=Config.TRESHOLD2, p=Config.PERCENTAGE, K=Config.K)
    print("Start Evaluation plan algorithm ..")
    successRate = EP.EvaluationPlanAlg()
    print("The success rate of this run is: " + str(successRate))


if __name__ == "__main__":
    main()
