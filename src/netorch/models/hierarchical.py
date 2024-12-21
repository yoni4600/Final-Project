# coding:utf-8
import numpy as np


class ConcatModel(object):

    def __init__(self, graph, dimension, Model, Coarsening, num_scales=4):
        self.original_graph = graph
        self.dimension = dimension
        self.Model = Model
        self.Coarsening = Coarsening
        self.num_scales = num_scales
        self.embeddings = np.zeros((graph.number_of_nodes(), 0))

    # def dimension_reduction(self, embeddings, dimension):
    #     return PCA(n_components=dimension).fit_transform(embeddings)

    def train(self):
        # Initialize the graph coarsening process
        coarsening = self.Coarsening(self.original_graph)

        # stopping condition to the recursive, If only one graph is returned perform recursive merging
        if len(coarsening.graphs) == 1:
            coarsening.recursive_merge()

        # Retrieve the list of coarsened graphs and their mappings to the original graph, keeps the order of supernodes
        graphs = coarsening.graphs
        mappings = coarsening.make_mappings_to_original_graph()

        # Prepare to select significant levels of graph coarsening based on node reduction
        prev_nodes = None
        indices = []
        for i, g in enumerate(graphs):
            # Reducing the number of nodes by at least 3% compared to the previous graph is considered significant.
            if prev_nodes is None or g.number_of_nodes()<prev_nodes*0.97:
                indices.append(i)
                prev_nodes = g.number_of_nodes()

        # Determine the number of scales and select indices to use for training
        num_scales = min(len(indices), self.num_scales)
        step = len(indices)/num_scales
        selected_indices = [int(np.ceil(i*step)) for i in range(num_scales)]
        selected_indices = [indices[i] for i in selected_indices]

        # Get the graphs and mappings for the selected indices
        train_graphs = [graphs[index] for index in selected_indices]
        train_mappings = [mappings[index] for index in selected_indices]

        # Set dimensions for each graph to be trained
        dimensions = [self.dimension for g in train_graphs]

        # Train models on each selected graph
        for i, (graph, mapping, dimension) in enumerate(zip(train_graphs, train_mappings, dimensions)):
            print('Training graph#{} #nodes={} #edges={}'.format(i, graph.number_of_nodes(), graph.number_of_edges()))
            # Reverse the mapping for embedding assignment
            rev_mapping = coarsening.reverse_mapping(mapping)
            # Initialize and train the embedding model
            model = self.Model(graph, dimension)
            results = model.train().get_embeddings()
            # Initialize array to hold embeddings for the original graph's nodes, creating the embedding matrix
            embeddings = np.ndarray(shape=(self.original_graph.number_of_nodes(), dimension))
            # Assign embeddings from the model to the corresponding nodes in the original graph
            for node, super_node in rev_mapping.items():
                embeddings[node,:] = results[super_node,:]
            # Concatenate new embeddings with existing embeddings
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=1)

        # self.embeddings = self.dimension_reduction(self.embeddings, self.dimension)
        
        return self
        
    def get_embeddings(self):
        # Return the concatenated embeddings from all scales
        return self.embeddings


            
