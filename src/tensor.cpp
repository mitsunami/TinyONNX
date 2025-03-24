#include "tensor.h"
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <iomanip>

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    size_t total_size = 1;
    for (int dim : shape) total_size *= dim;
    data_.resize(total_size);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data)
{
    // Validate that the shape matches the data size
    size_t total = 1;
    for (int dim : shape)
        total *= dim;

    assert(total == data.size() && "Shape does not match data size");
}

void Tensor::fillRandom() {
    for (auto& val : data_) {
        val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i != shape_.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Tensor data (first 10 values): ";
    size_t limit = std::min(data_.size(), static_cast<size_t>(10));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data_[i] << " ";
    }

    if (data_.size() > 10)
        std::cout << "...";

    std::cout << std::endl;
}

