# main_code.py
import networkx as nx
from netorch.dataset import load_edgelist, load_labels
from config import Config
import MainResearchPlan
DATASET_DIR = "datasets"
EDGES_TXT = ""
LABELS_TXT = ""


def main():
    # If EDGES_TXT or LABELS_TXT are empty, fall back to the default
    global EDGES_TXT, LABELS_TXT, DATASET_DIR

    if not EDGES_TXT:
        EDGES_TXT = "{}/{}_edgelist.txt".format(DATASET_DIR, "cora")
    if not LABELS_TXT:
        LABELS_TXT = "{}/{}_labels.txt".format(DATASET_DIR, "cora")

    print(f"Loading edgelist from: {EDGES_TXT}\n")
    print(f"Loading labels from: {LABELS_TXT}\n")

    g, labels = load_edgelist(EDGES_TXT), load_labels(LABELS_TXT)
    g = nx.convert_node_labels_to_integers(g)

    m = MainResearchPlan(
        g,
        d=Config.DIMENSION,
        t1=Config.TRESHOLD1,
        t2=Config.TRESHOLD2,
        p=Config.PERCENTAGE,
        K=Config.K
    )
    m.MainResearchPlanAlg()


if __name__ == "__main__":
    main()
