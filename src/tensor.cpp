#include "tensor.h"
#include <iostream>
#include <cstdlib>

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    size_t total_size = 1;
    for (int dim : shape) total_size *= dim;
    data_.resize(total_size);
}

void Tensor::fillRandom() {
    for (auto& val : data_) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
}

void Tensor::applyReLU() {
    for (auto& val : data_) {
        val = std::max(0.0f, val);
    }
}

void Tensor::print() const {
    std::cout << "Tensor data: ";
    for (size_t i = 0; i < data_.size() && i < 10; ++i)
        std::cout << data_[i] << " ";
    std::cout << "..." << std::endl;
}
