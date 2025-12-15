//
// Created by Matthew Ivanov on 14.12.2025.
//

#ifndef LAB2_BENCHMARK_H
#define LAB2_BENCHMARK_H

#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

struct BenchmarkResult {
    std::string function_name;
    std::string image_info;
    double time_μs{};
    double mp_per_sec{};
    double mb_per_sec{};
    int iterations{};
};

cv::Mat create_synthetic_hist_image(cv::Size size, std::mt19937& rng);
std::pair<cv::Mat, cv::Mat> create_synthetic_blend_images(cv::Size size, std::mt19937& rng);

bool validate_histogram(int random_seed);
bool validate_blend(int random_seed);

std::vector<BenchmarkResult> benchmark_histogram_synthetic(int iterations, int random_seed, uint8_t bins = 16);
std::vector<BenchmarkResult> benchmark_blend_synthetic(int iterations, int random_seed);
std::vector<BenchmarkResult> benchmark_real_images(
    int iterations,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& foreground_paths = {},
    uint8_t n_bins = 16
);

void generate_report(const std::vector<BenchmarkResult>& results, const std::string& filename = "benchmark_report.txt");
void run_all_benchmarks(int iterations, int random_seed, uint8_t bins = 16);

#endif //LAB2_BENCHMARK_H