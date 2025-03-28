#include "graph_utils.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>

void ComputationGraph::topologicalSort() {
    std::unordered_set<std::string> available;
    std::unordered_map<const GraphNode*, int> dependency_count;
    std::unordered_map<std::string, std::vector<const GraphNode*>> tensor_consumers;

    // Add all tensors that are already in graph.tensors (initializers or input)
    for (const auto& [name, _] : tensors) {
        available.insert(name);
    }
    available.insert("input");  // TODO: common input name — may vary

    // Count dependencies and build reverse edge map
    for (const auto& node : nodes) {
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

    // Enqueue nodes with no dependencies
    for (const auto& node : nodes) {
        if (dependency_count[&node] == 0) {
            ready.push(&node);
        }
    }

    while (!ready.empty()) {
        const GraphNode* node = ready.front();
        ready.pop();
        sorted_nodes.push_back(node);

        for (const auto& output : node->outputs) {
            for (const auto* consumer : tensor_consumers[output]) {
                if (--dependency_count[consumer] == 0) {
                    ready.push(consumer);
                }
            }
        }
    }


    if (sorted_nodes.size() != nodes.size()) {
        throw std::runtime_error("Cycle detected or missing inputs in graph");
    }

}

void ComputationGraph::printNodes() {
    std::cout << "🔁 Graph Order:\n";
    for (const auto node : nodes) {
        std::cout << "\033[1;32m- " << node.op_type << "\033[0m";
        if (!node.outputs.empty()) {
            std::cout << " : [" << node.inputs[0] << "] → [" << node.outputs[0] << "]";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void ComputationGraph::printSortedNodes() {
    std::cout << "🔁 Topological Node Execution Order:\n";
    for (const auto* node : sorted_nodes) {
        std::cout << "\033[1;32m- " << node->op_type << "\033[0m";
        if (!node->outputs.empty()) {
            std::cout << " : [" << node->inputs[0] << "] → [" << node->outputs[0] << "]";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

