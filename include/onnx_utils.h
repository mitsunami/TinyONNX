#pragma once
#include <string>
#include <onnx/onnx_pb.h>
#include "graph.h"

float getFloatAttr(const GraphNode* node, const std::string& name, float default_value);
int64_t getIntAttr(const GraphNode* node, const std::string& name, int64_t default_value);
std::vector<int> getIntListAttr(const GraphNode* node, const std::string& name);
