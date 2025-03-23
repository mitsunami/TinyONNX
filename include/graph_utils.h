#pragma once
#include "graph.h"

std::vector<const GraphNode*> topologicalSort(const ComputationGraph& graph);
void printTopologicalOrder(const std::vector<const GraphNode*>& sorted_nodes);
