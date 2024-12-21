#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#distutils: language=c++

from libcpp cimport bool
from libcpp.utility cimport pair
from necpp cimport Graph as CGraph
from necpp cimport Walker as CWalker, BiasedWalker as CBiasedWalker
from necpp cimport WindowSampling, SkipSampling
from necpp cimport ACOWalk

import networkx as nx

cdef class Graph:
    cdef CGraph c_graph

    @staticmethod
    def from_nx_graph(graph):
        g = Graph()
        for u, v in graph.edges:
            g.c_graph.AddEdge(u, v, graph[u][v]['weight'])
        return g

    def __cinit__(self):
        self.c_graph = CGraph()

    def add_edge(self, u, v, weight):
        self.c_graph.AddEdge(u, v, weight)

    def remove_edge(self, u, v, weight):
        self.c_graph.RemoveEdge(u, v)

    def edges(self):
        """Generator to yield edges."""
        cdef CGraph.EdgeView edge_view = self.c_graph.edges()
        cdef CGraph.EdgeView.Iterator edge_iter = edge_view.begin()
        cdef CGraph.EdgeView.Iterator edge_end = edge_view.end()
        cdef pair[int, int] edge

        while edge_iter != edge_end:
            edge = edge_iter.current()  # Use the current() helper to dereference
            yield (edge.first, edge.second)
            edge_iter.increment()  # Use increment() helper to move to the next item

cdef class Walker:
    cdef CWalker c_walker

    def __cinit__(self):
        self.c_walker = CWalker()

    def set_node_list(self, list nodes):
        self.c_walker.set_node_list(nodes)

    def set_transition_weights(self, int node, list neighbors, list weights):
        self.c_walker.SetTransitionWeights(node, neighbors, weights)

    def init_distributions_from_graph(self, Graph graph, bint weighted):
        self.c_walker.InitDistributionsFromGraph(graph.c_graph, weighted)
    
    def simulate_walk(self, size_t start_node, size_t walk_length):
        return self.c_walker.SimulateWalk(start_node, walk_length)

    def walk(self, size_t num_walks, size_t walk_length, size_t num_threads):
        return self.c_walker.Walk(num_walks, walk_length, num_threads)

cdef class BiasedWalker:
    cdef CBiasedWalker c_walker

    def __cinit__(self):
        self.c_walker = CBiasedWalker()

    def init_distributions_from_graph(self, Graph graph, double p, double q):
        self.c_walker.InitDistributionsFromGraph(graph.c_graph, p, q)
    
    def simulate_walk(self, size_t start_node, size_t walk_length):
        return self.c_walker.SimulateWalk(start_node, walk_length)

    def walk(self, size_t num_walks, size_t walk_length, size_t num_threads):
        return self.c_walker.Walk(num_walks, walk_length, num_threads)

def window_sampling(list sequences, size_t window_size, double down_sampling, bool shuffle):
    return WindowSampling(sequences, window_size, down_sampling, shuffle)

def skip_sampling(list sequences, size_t distance, double down_sampling, bool shuffle):
    return SkipSampling(sequences, distance, down_sampling, shuffle)

def aco_walk(Graph graph, size_t num_walks, size_t max_step, size_t num_iterations, double alpha, double evaporate, size_t num_threads):
    print("Initializing ACOWalk...")
    g = ACOWalk(graph.c_graph, num_walks, max_step, num_iterations, alpha, evaporate, num_threads)
    print("ACOWalk initialized successfully.")

    cdef list edge_list = []
    print("Converting EdgeView to Python list...")

    cdef CGraph.EdgeView edge_view = g.edges()
    cdef CGraph.EdgeView.Iterator edge_iter = edge_view.begin()
    cdef CGraph.EdgeView.Iterator edge_end = edge_view.end()
    cdef pair[int, int] edge

    while edge_iter != edge_end:
        print("Checking iterator...")
        edge = edge_iter.current()  # Call current() explicitly
        print(f"Edge: ({edge.first}, {edge.second})")
        edge_list.append((edge.first, edge.second))
        edge_iter.increment()  # Call increment() explicitly

    print("EdgeView iteration completed.")

    if len(edge_list) == 0:
        print("No edges found in graph.")
        return []

    print("Edges retrieved:", edge_list)

    phe = [(u, v, g.weight(u, v)) for u, v in edge_list]
    print("Weights computed:", phe)

    return phe