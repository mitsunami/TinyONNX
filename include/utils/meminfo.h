#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

inline size_t getPeakRSSinKB() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {  // Peak Resident Set Size
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;
            iss >> key >> value >> unit;
            return value;  // in KB
        }
    }
    return 0;
}

inline void printPeakRSS() {
    size_t peak_kb = getPeakRSSinKB();
    std::cout << "Peak memory (RSS): " << peak_kb << " KB ≈ " << peak_kb / 1024.0 << " MB" << std::endl;
}
