#pragma once
#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    Timer(const std::string& label = "") : label_(label) {
#ifdef ENABLE_TIMING
        start_ = std::chrono::high_resolution_clock::now();
#endif
    }

    ~Timer() {
#ifdef ENABLE_TIMING
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::cout << label_ << ": " << ms << " ms" << std::endl;
#endif
    }

private:
    std::string label_;
#ifdef ENABLE_TIMING
    std::chrono::high_resolution_clock::time_point start_;
#endif
};
