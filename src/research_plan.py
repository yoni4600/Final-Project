# coding:utf-8

from netorch.coarsening.aco import ACOCoarsening
from netorch.models.walkbased import Node2Vec
from netorch.models.hierarchical import MLNE


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
            # TODO: RemoveEdges(g, Percentage p)

            # loads the model with the embedding method and coarsening method
            model = MLNE(
                graph=g,
                dimension=128,
                Model=lambda graph, dimension: Node2Vec(graph, dimension=dimension, batch_size=5000, iterations=3, p=1,
                                                        q=1),
                Coarsening=lambda graph: ACOCoarsening(graph, phe_power=.5, iterations=2),
            )
            # start the process and receive the embedding matrix
            embedding_matrix = model.train().get_embeddings()

            # TODO: similarity_matrix = CalculateCosineSim(embedding_matrix, Treshold t1, )
            # similarity_matrix_list.append(similarity_matrix)

        # TODO: M_Stat = calculateStatistics(similarity_matrix_list)

        # TODO: G_R = refineGraph(g, M_Stat, Treshold t2)
        # TODO: return G_R

        # result = evaluate({lookup.index_to_label(index):embedding[index] for index in range(g.number_of_nodes())}, labels, clf_ratio=0.5)
        # print(result)
