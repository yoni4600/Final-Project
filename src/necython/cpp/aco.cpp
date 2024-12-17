#include "aco.hpp"

#include <vector>
#include <unordered_map>
#include <iostream>

namespace network_embedding {

using std::size_t;
using std::vector;
using std::unordered_map;

static void CalculateLoopPheromone(
        const vector<NodeList> & sequences,
        size_t max_step,
        size_t num_threads,
        Graph & pheromone) {
    for (const auto & sequence : sequences) {
        for (size_t i = 0; i < sequence.size(); i++) {
            for (size_t len = 1; len < max_step && i+len < sequence.size(); len++) {
                if (sequence[i] == sequence[i+len]) {
                    for (size_t k = i; k < i+len; k++) {
                        const Node & u = sequence[k];
                        const Node & v = sequence[k+1];
                        pheromone.SetEdgeWeight(u, v, pheromone.weight(u, v) + 1.0/len);
                    }
                    break;
                }
            }
        }
    }
}

static void CalculatePheromoneWithLabel(
        const vector<NodeList> & sequences,
        const unordered_map<Node, vector<int>> labels,
        size_t max_step,
        size_t num_threads,
        Graph & pheromone) {
    for (const auto & sequence : sequences) {
        for (size_t i = 0; i < sequence.size(); i++) {
            for (size_t len = 1; len < max_step && i+len < sequence.size(); len++) {
                if (labels.find(sequence[i+len]) != labels.end()) {
                    for (size_t k = i; k < i+len; k++) {
                        const Node & u = sequence[k];
                        const Node & v = sequence[k+1];
                        pheromone.SetEdgeWeight(u, v, pheromone.weight(u, v) + 1.0/len);
                    }
                    break;
                }
            }
        }
    }
}

Graph ACOWalk(
        const Graph & graph,
        size_t num_walks,
        size_t max_step,
        size_t num_iterations,
        double alpha,
        double evaporate,
        size_t num_threads) {

    Graph g(graph);
    Graph pheromone(graph);
    Graph total_pheromone(graph);
    Graph mixed(graph);


    // Debug edge initialization
    if (g.number_of_edges() == 0) {
        std::cerr << "Error: Input graph has no edges!" << std::endl;
        return total_pheromone;
    }

    for (size_t i = 0; i < num_iterations; i++) {

        // Initialize walker and simulate walks
        Walker walker;
        walker.InitDistributionsFromGraph(mixed, true);

        auto sequences = walker.Walk(num_walks, max_step, num_threads);

        // Debug sequences
        for (size_t j = 0; j < sequences.size(); ++j) {
            const auto &sequence = sequences[j];
            for (const auto &node : sequence) {
            }
        }

        // Reset pheromone weights
        for (const auto & edge : g.edges()) {
            pheromone.SetEdgeWeight(edge, 0.0);
        }

        // Calculate loop pheromone
        CalculateLoopPheromone(sequences, max_step, num_threads, pheromone);

        // Update total pheromone
        for (const auto & edge : g.edges()) {
            double old_weight = total_pheromone.weight(edge);
            double new_weight = old_weight * (1.0 - evaporate) + pheromone.weight(edge);
            total_pheromone.SetEdgeWeight(edge, new_weight);
        }

        // Update mixed graph weights
        for (const auto & edge : g.edges()) {
            double old_weight = mixed.weight(edge);
            double new_weight = g.weight(edge) * pow(total_pheromone.weight(edge), alpha);
            mixed.SetEdgeWeight(edge, new_weight);
        }
    }

    // Final pheromone power adjustment
    for (const auto & edge : g.edges()) {
        double old_weight = total_pheromone.weight(edge);
        double new_weight = pow(old_weight, alpha);
        total_pheromone.SetEdgeWeight(edge, new_weight);
    }

    return total_pheromone;
}

Graph ACOWalkWithLabel(
        const Graph & graph,
        const unordered_map<Node, vector<int>> labels,
        size_t num_walks,
        size_t max_step,
        size_t num_iterations,
        double alpha,
        double evaporate,
        size_t num_threads) {

    Graph g(graph);
    Graph pheromone(graph);
    Graph total_pheromone(graph);
    Graph mixed(graph);

    std::cout << "Debug: Printing edges in graph..." << std::endl;
    for (const auto & edge : g.edges()) {
        std::cout << "Edge: (" << edge.first << ", " << edge.second << ")" << std::endl;
    }

    for (size_t i = 0; i < num_iterations; i++) {
        Walker walker;
        walker.InitDistributionsFromGraph(mixed, true);
        auto sequences = walker.Walk(num_walks, 80, num_threads);

        for (const auto & edge : g.edges()) {
            pheromone.SetEdgeWeight(edge, 0.0);
        }

        CalculateLoopPheromone(sequences, max_step, num_threads, pheromone);
        CalculatePheromoneWithLabel(sequences, labels, max_step, num_threads, pheromone);

        for (const auto & edge : g.edges()) {
            total_pheromone.SetEdgeWeight(edge, total_pheromone.weight(edge) * (1.0-evaporate) + pheromone.weight(edge));
        }

        for (const auto & edge : g.edges()) {
            mixed.SetEdgeWeight(edge, g.weight(edge) * pow(total_pheromone.weight(edge), alpha));
        }
    }

    return total_pheromone;
}

};
