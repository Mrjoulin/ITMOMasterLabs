//
// Created by Matthew Ivanov on 29.09.2025.
//

#ifndef LAB1_BENCHMARK_H
#define LAB1_BENCHMARK_H


#include "hashmap.h"
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>

class Benchmark {
public:
    struct BenchmarkResult {
        double mean{};
        double median{};
        double min{};
        double max{};
        size_t operations{};
        std::vector<double> percentiles;
    };

    static constexpr size_t CYCLES = 100;
    static constexpr size_t MEASUREMENT_ITERATIONS = 10000;
    static constexpr double PERCENTILES[] = {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0};

    // Benchmark scenarios
    static BenchmarkResult create_avg_result(const std::vector<BenchmarkResult> &results);
    static void benchmark_case(std::vector<int>& keys, const std::string& scenario);
    static void benchmark_optimal_case(size_t iterations);
    static void benchmark_worst_case(size_t iterations);
    static void benchmark_random_case(size_t iterations);
    static void benchmark_all_scenarios(size_t iterations);
    static void benchmark_all_scenarios();

    // Individual operation benchmarks
    static BenchmarkResult create_benchmark_result(std::vector<double>& latencies);
    static BenchmarkResult benchmark_insertions(
        HashMap& dict, const std::vector<int>& keys, const std::string& value,
        const std::optional<std::string>& file_name);
    static BenchmarkResult benchmark_finds(
        const HashMap& dict, const std::vector<int>& keys,
        const std::optional<std::string>& file_name);
    static BenchmarkResult benchmark_deletions(
        HashMap& dict, const std::vector<int>& keys,
        const std::optional<std::string>& file_name);

    // Data generation
    static std::vector<int> generate_optimal_keys(size_t count);
    static std::vector<int> generate_worst_case_keys(size_t count);
    static std::vector<int> generate_random_keys(size_t count);

    // Utility functions
    static void write_latencies(const std::vector<double>& latencies, const std::string& file_name);
    static void print_result(const BenchmarkResult& result, const std::string& operation, const std::string& scenario);
};


#endif //LAB1_BENCHMARK_H