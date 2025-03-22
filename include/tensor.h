#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape);

    void fillRandom();
    void applyReLU();
    void print() const;
    
    std::vector<int> shape() const { return shape_; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

private:
    std::vector<int> shape_;
    std::vector<float> data_;
};

#endif // TENSOR_H
