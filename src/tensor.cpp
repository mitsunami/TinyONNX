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

// Transpose in-place: PyTorch-style weights [OC, IC, KH, KW] → XNNPACK-style [OC, KH, KW, IC]
void Tensor::reorderOIHWtoOHWI() {
    if (shape_.size() != 4)
        throw std::invalid_argument("Weight tensor must be 4D");

    int OC = shape_[0];
    int IC = shape_[1];
    int KH = shape_[2];
    int KW = shape_[3];

    std::vector<float> new_data(OC * KH * KW * IC);

    for (int oc = 0; oc < OC; ++oc) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                for (int ic = 0; ic < IC; ++ic) {
                    int src_index = (((oc * IC + ic) * KH + kh) * KW) + kw;
                    int dst_index = (((oc * KH + kh) * KW + kw) * IC) + ic;
                    new_data[dst_index] = data_[src_index];
                }
            }
        }
    }

    data_ = std::move(new_data);
    shape_ = {OC, KH, KW, IC};
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

