// logger.h
#pragma once
#include <iostream>
#include <sstream>

enum LogLevel {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_NONE
};

class Logger {
public:
    static Logger& instance() {
        static Logger instance_;
        return instance_;
    }

    void setLevel(LogLevel level) { current_level_ = level; }

    template<typename... Args>
    void debug(Args&&... args) const {
        log(LOG_LEVEL_DEBUG, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(Args&&... args) const {
        log(LOG_LEVEL_INFO, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warning(Args&&... args) const {
        log(LOG_LEVEL_WARNING, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(Args&&... args) const {
        log(LOG_LEVEL_ERROR, std::forward<Args>(args)...);
    }

private:
    LogLevel current_level_ = LOG_LEVEL_INFO;

    template<typename... Args>
    void log(LogLevel level, Args&&... args) const {
        if (current_level_ <= level) {
            std::ostringstream oss;
            (oss << ... << args);  // fold-expression (C++17), concatenates args
            if (level >= LOG_LEVEL_WARNING)
                std::cerr << oss.str() << std::endl;
            else
                std::cout << oss.str() << std::endl;
        }
    }
};
