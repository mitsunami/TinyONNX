#include "graph_utils.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>

std::vector<const GraphNode*> topologicalSort(const ComputationGraph& graph) {
    std::unordered_set<std::string> available;
    std::unordered_map<const GraphNode*, int> dependency_count;
    std::unordered_map<std::string, std::vector<const GraphNode*>> tensor_consumers;

    // Add all tensors that are already in graph.tensors (initializers or input)
    for (const auto& [name, _] : graph.tensors) {
        available.insert(name);
    }
    available.insert("input");  // TODO: common input name — may vary

    // Count dependencies and build reverse edge map
    for (const auto& node : graph.nodes) {
        int count = 0;
        for (const auto& input : node.inputs) {
            if (!available.count(input)) {
                count++;
                tensor_consumers[input].push_back(&node);
            }
        }
        dependency_count[&node] = count;
    }

    std::queue<const GraphNode*> ready;
    std::vector<const GraphNode*> sorted;

    // Enqueue nodes with no dependencies
    for (const auto& node : graph.nodes) {
        if (dependency_count[&node] == 0) {
            ready.push(&node);
        }
    }

    while (!ready.empty()) {
        const GraphNode* node = ready.front();
        ready.pop();
        sorted.push_back(node);

        for (const auto& output : node->outputs) {
            for (const auto* consumer : tensor_consumers[output]) {
                if (--dependency_count[consumer] == 0) {
                    ready.push(consumer);
                }
            }
        }
    }

    if (sorted.size() != graph.nodes.size()) {
        throw std::runtime_error("Cycle detected or missing inputs in graph");
    }

    //printTopologicalOrder(sorted);

    return sorted;
}

void printTopologicalOrder(const std::vector<const GraphNode*>& sorted_nodes) {
    std::cout << "🔁 Topological Node Execution Order:\n";
    for (const auto* node : sorted_nodes) {
        std::cout << "\033[1;32m- " << node->op_type << "\033[0m";
        if (!node->outputs.empty()) {
            std::cout << " → [" << node->outputs[0] << "]";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}
