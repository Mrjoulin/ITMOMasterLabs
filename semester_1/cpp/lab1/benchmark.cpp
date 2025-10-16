//
// Created by Matthew Ivanov on 29.09.2025.
//

#include "benchmark.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <random>


// Generate keys that minimize collisions (well-distributed)
std::vector<int> Benchmark::generate_optimal_keys(const size_t count) {
    std::vector<int> keys;
    keys.reserve(count);

    HashMap test_dict;
    while (test_dict.capacity() < count * 10 / 7) {
        test_dict.resize_prime(test_dict.capacity() * 2 + 1);
    }

    std::vector<bool> hashes(test_dict.capacity(), false);

    size_t offset = 0;
    for (int i = 0; i < count + offset; ++i) {
        auto key_hash = test_dict.hash(i);
        if (!hashes[key_hash]) {
            keys.push_back(i);
            hashes[key_hash] = true;
        } else {
            ++offset;
        }
    }
    return keys;
}

// Generate keys that maximize collisions
std::vector<int> Benchmark::generate_worst_case_keys(const size_t count) {
    std::vector<int> keys;
    keys.reserve(count);

    HashMap test_dict;
    while (test_dict.capacity() < count * 10 / 7) {
        test_dict.resize_prime(test_dict.capacity() * 2 + 1);
    }
    std::vector<bool> hashes(test_dict.capacity(), false);

    // insert first half as is
    for (int i = -static_cast<int>(count)/4; i <= static_cast<int>(count)/4; ++i) {
        keys.push_back(i);
        auto key_hash = test_dict.hash(i);
        while (hashes[key_hash]) {
            ++key_hash;
            if (key_hash >= test_dict.capacity()) key_hash = 0;
        }
        hashes[key_hash] = true;
    }

    // second half with max collisions
    int key = static_cast<int>(count) / 4 + 1;
    while (keys.size() < count) {
        for (int iter = 0; iter < 2; ++iter) {
            auto key_hash = test_dict.hash(key);
            if (hashes[key_hash]) {
                keys.push_back(key);
                while (hashes[key_hash]) {
                    ++key_hash;
                    if (key_hash >= test_dict.capacity()) key_hash = 0;
                }
                hashes[key_hash] = true;
            }
            key = -key;
            if (keys.size() >= count) break;
        }
        ++key;
    }

    return keys;
}

// Generate completely random keys
std::vector<int> Benchmark::generate_random_keys(const size_t count) {
    std::vector<int> keys;
    keys.reserve(count);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dist(-1000000, 1000000);

    for (size_t i = 0; i < count; ++i) {
        keys.push_back(dist(gen));
    }

    return keys;
}

void Benchmark::write_latencies(const std::vector<double>& latencies, const std::string& file_name) {
    std::ofstream f;
    f.open(file_name);
    for (auto& x : latencies) {
        f << x << std::endl;
    }
    f.close();
}

Benchmark::BenchmarkResult Benchmark::create_benchmark_result(std::vector<double>& latencies) {
    std::ranges::sort(latencies);

    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    std::vector<double> percentiles;

    for (const double p : PERCENTILES) {
        const auto index = static_cast<size_t>(p * static_cast<double>(latencies.size() - 1));
        percentiles.push_back(latencies[index]);
    }

    const BenchmarkResult result = {
        .mean = sum / static_cast<double>(latencies.size()),
        .median = latencies[latencies.size() / 2],
        .min = latencies.front(),
        .max = latencies.back(),
        .operations = latencies.size(),
        .percentiles = percentiles
    };

    return result;
}


Benchmark::BenchmarkResult Benchmark::benchmark_insertions(
        HashMap& dict, const std::vector<int>& keys, const std::string& value,
        const std::optional<std::string>& scenario) {
    std::vector<double> latencies;
    latencies.reserve(keys.size());

    // Actual measurement
    for (size_t i = 0; i < keys.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        dict.insert(keys[i], value);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::nano>(end - start).count();
        latencies.push_back(duration);
    }

    if (scenario.has_value())
        write_latencies(latencies, scenario.value() + "_insert.txt");

    return create_benchmark_result(latencies);
}

Benchmark::BenchmarkResult Benchmark::benchmark_finds(
        const HashMap& dict, const std::vector<int>& keys,
        const std::optional<std::string>& scenario) {
    std::vector<double> latencies;
    latencies.reserve(keys.size());

    // Actual measurement
    for (size_t i = 0; i < keys.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = dict.find(keys[i]);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::nano>(end - start).count();
        latencies.push_back(duration);
    }

    if (scenario.has_value())
        write_latencies(latencies, scenario.value() + "_find.txt");

    return create_benchmark_result(latencies);
}

Benchmark::BenchmarkResult Benchmark::benchmark_deletions(
        HashMap& dict, const std::vector<int>& keys,
        const std::optional<std::string>& scenario) {
    std::vector<double> latencies;
    latencies.reserve(keys.size());

    // Actual measurement
    for (size_t i = 0; i < keys.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        dict.remove(keys[i]);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::nano>(end - start).count();
        latencies.push_back(duration);
    }

    if (scenario.has_value())
        write_latencies(latencies, scenario.value() + "_delete.txt");

    return create_benchmark_result(latencies);
}

void Benchmark::print_result(const BenchmarkResult& result, const std::string& operation, const std::string& scenario) {
    std::cout << "\n=== " << scenario << " - " << operation << " ===" << std::endl;
    std::cout << "Operations: " << result.operations << std::endl;
    std::cout << "Mean: " << std::fixed << std::setprecision(2) << result.mean << " ns" << std::endl;
    std::cout << "Median: " << result.median << " ns" << std::endl;
    std::cout << "Min: " << result.min << " ns" << std::endl;
    std::cout << "Max: " << result.max << " ns" << std::endl;
    std::cout << "Latency Distribution (ns):" << std::endl << "Percentile: ";
    for (const double p : PERCENTILES) {
        std::cout << std::setw(10) << static_cast<int>(p * 100);
    }
    std::cout << std::endl << "Time (ns):  ";
    for (const double val : result.percentiles) {
        std::cout << std::setw(10) << val;
    }
}

Benchmark::BenchmarkResult Benchmark::create_avg_result(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return {};
    if (results.size() == 1) { return results[0]; }

    std::vector<double> percentiles;
    for (const double p : PERCENTILES) percentiles.push_back(0);

    std::vector<double> final_results(4, 0);

    for (auto& result : results) {
        final_results[0] += result.mean;
        final_results[1] += result.median;
        final_results[2] += result.min;
        final_results[3] += result.max;
        for (int i = 0; i < result.percentiles.size(); ++i) {
            percentiles[i] += result.percentiles[i];
        }
    }

    for (auto& p : percentiles) { p /= static_cast<double>(results.size()); }

    return {
        .mean = final_results[0] / static_cast<double>(results.size()),
        .median = final_results[1] / static_cast<double>(results.size()),
        .min = final_results[2] / static_cast<double>(results.size()),
        .max = final_results[3] / static_cast<double>(results.size()),
        .operations = results[0].operations,
        .percentiles = percentiles,
    };
}


void Benchmark::benchmark_case(std::vector<int>& keys, const std::string& scenario) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK: " << scenario << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::vector<BenchmarkResult> insert_results;
    std::vector<BenchmarkResult> find_results;
    std::vector<BenchmarkResult> delete_results;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 0; i < CYCLES; ++i) {
        // Reshuffle every time
        std::ranges::shuffle(keys, gen);
        HashMap dict;

        std::optional<std::string> write_scenario = (i == CYCLES - 1) ? std::optional(scenario) : std::nullopt;
        const auto insert_result = benchmark_insertions(dict, keys, "test_value", write_scenario);
        const auto find_result = benchmark_finds(dict, keys, write_scenario);
        const auto delete_result = benchmark_deletions(dict, keys, write_scenario);

        insert_results.push_back(insert_result);
        find_results.push_back(find_result);
        delete_results.push_back(delete_result);
    }

    print_result(create_avg_result(insert_results), "INSERT", scenario);
    print_result(create_avg_result(find_results), "FIND", scenario);
    print_result(create_avg_result(delete_results), "DELETE", scenario);
}


void Benchmark::benchmark_optimal_case(const size_t iterations) {
    auto keys = generate_optimal_keys(iterations);
    benchmark_case(keys, "Optimal Case");
}

void Benchmark::benchmark_worst_case(const size_t iterations) {
    auto keys = generate_worst_case_keys(iterations);
    benchmark_case(keys, "Worst Case");
}

void Benchmark::benchmark_random_case(const size_t iterations) {
    auto keys = generate_random_keys(iterations);
    benchmark_case(keys, "Random Case");
}

void Benchmark::benchmark_all_scenarios(const size_t iterations) {
    std::cout << "STARTING COMPREHENSIVE HASHMAP BENCHMARK" << std::endl;
    std::cout << "Measurement iterations: " << iterations << std::endl;

    benchmark_optimal_case(iterations);
    benchmark_worst_case(iterations);
    benchmark_random_case(iterations);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void Benchmark::benchmark_all_scenarios() {
    benchmark_all_scenarios(MEASUREMENT_ITERATIONS);
}