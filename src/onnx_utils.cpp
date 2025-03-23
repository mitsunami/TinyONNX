#include "onnx_utils.h"

float getFloatAttr(const GraphNode* node, const std::string& name, float default_value) {
    for (const auto& attr : node->attributes) {
        if (attr.name() == name && attr.has_f())
            return attr.f();
    }
    return default_value;
}

int64_t getIntAttr(const GraphNode* node, const std::string& name, int64_t default_value) {
    for (const auto& attr : node->attributes) {
        if (attr.name() == name && attr.has_i())
            return attr.i();
    }
    return default_value;
}

std::vector<int> getIntListAttr(const GraphNode* node, const std::string& name) {
    for (const auto& attr : node->attributes) {
        if (attr.name() == name && attr.ints_size() > 0) {
            std::vector<int> result(attr.ints().begin(), attr.ints().end());
            return result;
        }
    }
    return {};
}
