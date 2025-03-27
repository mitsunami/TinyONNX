#pragma once
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    void fillRandom();
    void reorderOIHWtoOHWI();
    void print() const;

    std::vector<int> shape() const { return shape_; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

private:
    std::vector<int> shape_;
    std::vector<float> data_;
};
