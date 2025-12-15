#include "benchmark.h"

#include "blend.h"
#include "hist.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>


// ==================== VALIDATION FUNCTIONS ====================

cv::Mat create_synthetic_hist_image(const cv::Size size, std::mt19937& rng) {
    cv::Mat image(size, CV_8UC1);
    // Fill with random values
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            image.at<uint8_t>(i, j) = dist(rng);
        }
    }
    return image;
}

std::pair<cv::Mat, cv::Mat> create_synthetic_blend_images(const cv::Size size, std::mt19937& rng) {
    cv::Mat foreground(size, CV_8UC4);
    cv::Mat background(size, CV_8UC4);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    // Fill with random data
    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            auto& fg = foreground.at<cv::Vec4b>(i, j);
            auto& bg = background.at<cv::Vec4b>(i, j);
            for (int k = 0; k < 4; ++k) {
                fg[k] = dist(rng);
                bg[k] = dist(rng);
            }
        }
    }
    return {foreground, background};
}


bool validate_histogram(const int random_seed) {
    std::cout << "\n=== Validating Histogram Functions ===\n";

    // Test with different image sizes and bin counts
    std::vector<std::pair<cv::Size, uint8_t>> test_cases = {
        {cv::Size(640, 480), 32},
        {cv::Size(1280, 720), 64},
        {cv::Size(1920, 1080), 32},
        {cv::Size(100, 100), 16},
        {cv::Size(7, 13), 8}  // Odd dimensions to test edge cases
    };

    bool all_passed = true;
    std::mt19937 rng(random_seed);

    for (const auto& [size, bins] : test_cases) {
        cv::Mat image = create_synthetic_hist_image(size, rng);

        auto hist_naive = histogram_naive(image, bins);
        auto hist_simd = histogram_simd(image, bins);

        bool passed = true;
        if (hist_naive.size() != hist_simd.size()) {
            std::cout << "FAIL: Size mismatch for " << size << " bins=" << int(bins)
                     << " (naive=" << hist_naive.size() << ", simd=" << hist_simd.size() << ")\n";
            passed = false;
        } else {
            for (size_t i = 0; i < hist_naive.size(); ++i) {
                if (hist_naive[i] != hist_simd[i]) {
                    std::cout << "FAIL: Bin " << i << " mismatch for " << size
                             << " bins=" << int(bins) << " (naive=" << hist_naive[i]
                             << ", simd=" << hist_simd[i] << ")\n";
                    passed = false;
                    break;
                }
            }
        }

        if (passed) {
            std::cout << "PASS: " << size << " bins=" << int(bins) << "\n";
        }
        all_passed &= passed;
    }

    return all_passed;
}

bool validate_blend(const int random_seed) {
    std::cout << "\n=== Validating Blend Functions ===\n";

    std::vector<cv::Size> test_sizes = {
        cv::Size(640, 480),
        cv::Size(1280, 720),
        cv::Size(1920, 1080),
        cv::Size(101, 103)  // Odd dimensions
    };

    bool all_passed = true;
    std::mt19937 rng(random_seed);

    for (const auto& size : test_sizes) {
        auto [foreground, background] = create_synthetic_blend_images(size, rng);

        cv::Mat result_naive = blend_naive(foreground, background);
        cv::Mat result_simd = blend_simd(foreground, background);

        bool passed = true;
        if (result_naive.size() != result_simd.size() || result_naive.type() != result_simd.type()) {
            std::cout << "FAIL: Size/type mismatch for " << size << "\n";
            passed = false;
        } else {
            cv::Mat diff;
            cv::absdiff(result_naive, result_simd, diff);
            double max_diff;
            cv::minMaxLoc(diff, nullptr, &max_diff);

            // Allow small floating-point differences (if using float operations)
            if (max_diff > 1) {  // Allow 1 unit difference for integer operations
                std::cout << "FAIL: Pixel mismatch for " << size
                         << " max_diff=" << max_diff << "\n";
                passed = false;
            }
        }

        if (passed) {
            std::cout << "PASS: " << size << "\n";
        }
        all_passed &= passed;
    }

    return all_passed;
}

// ==================== BENCHMARKING FRAMEWORK ====================


class Benchmark {
private:
    using Clock = std::chrono::high_resolution_clock;

public:
    template<typename Func, typename... Args>
    static BenchmarkResult run_benchmark(
            const std::string& name,
            const cv::Mat& image,
            Func func,
            const int iterations,
            Args&&... args
    ) {
        // Warm-up
        for (int i = 0; i < 3; ++i) {
            func(std::forward<Args>(args)...);
        }

        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            func(std::forward<Args>(args)...);
        }
        const auto end = Clock::now();

        const double all_time_μs = std::chrono::duration<double, std::micro>(end - start).count();
        const double time_μs_per_iter = all_time_μs / iterations;

        const int total_pixels = image.rows * image.cols;
        const int total_bytes = total_pixels * static_cast<int>(image.elemSize());

        BenchmarkResult result;
        result.function_name = name;
        result.image_info = std::to_string(image.cols) + "x" + std::to_string(image.rows);
        result.time_μs = time_μs_per_iter;
        result.mp_per_sec = total_pixels / time_μs_per_iter;
        result.mb_per_sec = total_bytes / time_μs_per_iter;
        result.iterations = iterations;

        return result;
    }
};

// ==================== SYNTHETIC DATA BENCHMARKS ====================

std::vector<BenchmarkResult> benchmark_histogram_synthetic(
        const int iterations, const int random_seed, const uint8_t bins
) {
    std::cout << "\n=== Benchmarking Histogram on Synthetic Data ===\n";

    std::vector<cv::Size> test_sizes = {
        cv::Size(640, 480),      // VGA
        cv::Size(1280, 720),     // HD
        cv::Size(1920, 1080),    // Full HD
        cv::Size(3840, 2160),    // 4K
        cv::Size(512, 512),      // Square
        cv::Size(1000, 1000)     // Large square
    };

    std::vector<BenchmarkResult> results;
    std::mt19937 rng(random_seed);

    for (const auto& size : test_sizes) {
        cv::Mat image = create_synthetic_hist_image(size, rng);

        std::cout << "Testing " << size << " ("
                  << size.width * size.height / 1000000.0 << " MP)...\n";

        auto naive_result = Benchmark::run_benchmark(
            "histogram_naive", image, [&]() { histogram_naive(image, bins); }, iterations);
        results.push_back(naive_result);

        auto simd_result = Benchmark::run_benchmark(
            "histogram_simd", image, [&]() { histogram_simd(image, bins); }, iterations);
        results.push_back(simd_result);

        double speedup = naive_result.time_μs / simd_result.time_μs;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    return results;
}

std::vector<BenchmarkResult> benchmark_blend_synthetic(const int iterations, const int random_seed) {
    std::cout << "\n=== Benchmarking Blend on Synthetic Data ===\n";

    std::vector<cv::Size> test_sizes = {
        cv::Size(640, 480),      // VGA
        cv::Size(1280, 720),     // HD
        cv::Size(1920, 1080),    // Full HD
        cv::Size(3840, 2160),    // 4K
        cv::Size(512, 512),      // Square
        cv::Size(2000, 2000)     // Large square
    };

    std::vector<BenchmarkResult> results;
    std::mt19937 rng(random_seed);

    for (const auto& size : test_sizes) {
        auto [foreground, background] = create_synthetic_blend_images(size, rng);

        std::cout << "Testing " << size << " ("
                  << size.width * size.height / 1000000.0 << " MP)...\n";

        auto naive_result = Benchmark::run_benchmark(
            "blend_naive", foreground,
            [&]() { blend_naive(foreground, background); },
            iterations
        );
        results.push_back(naive_result);

        auto simd_result = Benchmark::run_benchmark(
            "blend_simd", foreground,
            [&]() { blend_simd(foreground, background); },
            iterations
        );
        results.push_back(simd_result);

        double speedup = naive_result.time_μs / simd_result.time_μs;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    return results;
}

// ==================== REAL IMAGE BENCHMARKS ====================

std::vector<BenchmarkResult> benchmark_real_images(
    const int iterations,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& foreground_paths,
    const uint8_t n_bins
) {
    std::vector<BenchmarkResult> results;

    if (!foreground_paths.empty()) {
        std::cout << "\n=== Benchmarking Blend on Real Images ===\n";

        for (size_t i = 0; i < std::min(image_paths.size(), foreground_paths.size()); ++i) {
            cv::Mat foreground = cv::imread(foreground_paths[i], cv::IMREAD_UNCHANGED);
            cv::Mat background = cv::imread(image_paths[i], cv::IMREAD_UNCHANGED);

            if (foreground.empty() || background.empty()) {
                std::cerr << "Failed to load images: "
                         << foreground_paths[i] << " or " << image_paths[i] << "\n";
                continue;
            }

            // Convert to BGRA if needed
            if (foreground.channels() == 3) {
                cv::cvtColor(foreground, foreground, cv::COLOR_BGR2BGRA);
            }
            if (background.channels() == 3) {
                cv::cvtColor(background, background, cv::COLOR_BGR2BGRA);
            }

            // Resize background to match foreground if sizes differ
            if (foreground.size() != background.size()) {
                cv::resize(background, background, foreground.size());
            }

            std::cout << "Testing " << foreground.cols << "x" << foreground.rows
                     << " (" << foreground_paths[i] << ")...\n";

            auto naive_result = Benchmark::run_benchmark(
                "blend_naive", foreground,
                [&]() { blend_naive(foreground, background); },
                iterations
            );
            results.push_back(naive_result);

            auto simd_result = Benchmark::run_benchmark(
                "blend_simd", foreground,
                [&]() { blend_simd(foreground, background); },
                iterations
            );
            results.push_back(simd_result);

            double speedup = naive_result.time_μs / simd_result.time_μs;
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
    }
    std::cout << "\n=== Benchmarking Histogram on Real Images ===\n";

    for (const auto& path : image_paths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << "\n";
            continue;
        }

        std::cout << "Testing " << image.cols << "x" << image.rows << " (" << path << ")...\n";

        auto naive_result = Benchmark::run_benchmark(
            "histogram_naive", image,
            [&]() { histogram_naive(image, n_bins); },
            iterations
        );
        results.push_back(naive_result);

        auto simd_result = Benchmark::run_benchmark(
            "histogram_simd", image,
            [&]() { histogram_simd(image, n_bins); },
            iterations
        );
        results.push_back(simd_result);

        double speedup = naive_result.time_μs / simd_result.time_μs;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    return results;
}

// ==================== REPORT GENERATION ====================

void generate_report(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream report(filename);

    if (!report.is_open()) {
        std::cerr << "Failed to open report file: " << filename << "\n";
        return;
    }

    report << "=== SIMD BENCHMARK REPORT ===\n\n";
    report << "Generated: " << __DATE__ << " " << __TIME__ << "\n\n";

    report << std::left << std::setw(20) << "Function"
           << std::setw(15) << "Image Size"
           << std::setw(15) << "Time (μs)"
           << std::setw(15) << "MP/sec"
           << std::setw(15) << "MB/sec"
           << std::setw(10) << "Iterations" << "\n";

    report << std::string(90, '-') << "\n";

    for (const auto& result : results) {
        report << std::left << std::setw(20) << result.function_name
               << std::setw(15) << result.image_info
               << std::fixed << std::setprecision(3)
               << std::setw(15) << result.time_μs
               << std::setw(15) << result.mp_per_sec
               << std::setw(15) << result.mb_per_sec
               << std::setw(10) << result.iterations << "\n";
    }

    // Calculate average speedups
    double hist_speedup[3] = {0, 0, 0}; // sum, max, cnt
    double blend_speedup[3] = {0, 0, 0};

    for (size_t i = 0; i + 1 < results.size(); i += 2) {
        double speedup = results[i].time_μs / results[i + 1].time_μs;
        if (results[i].function_name.find("histogram") != std::string::npos) {
            hist_speedup[0] += speedup;
            hist_speedup[1] = std::max(hist_speedup[1], speedup);
            hist_speedup[2]++;
        } else if (results[i].function_name.find("blend") != std::string::npos) {
            blend_speedup[0] += speedup;
            blend_speedup[1] = std::max(blend_speedup[1], speedup);
            blend_speedup[2]++;
        }
    }

    report << "\n=== SUMMARY ===\n";
    if (hist_speedup[2] > 0) {
        report << "Average histogram speedup: "
               << std::fixed << std::setprecision(2)
               << (hist_speedup[0] / hist_speedup[2]) << "x\n";
        report << "Max histogram speedup: "
               << std::fixed << std::setprecision(2)
               << hist_speedup[1] << "x\n\n";
    }
    if (blend_speedup[2] > 0) {
        report << "Average blend speedup: "
               << std::fixed << std::setprecision(2)
               << (blend_speedup[0] / blend_speedup[2]) << "x\n";
        report << "Max blend speedup: "
               << std::fixed << std::setprecision(2)
               << blend_speedup[1] << "x\n";;
    }

    report.close();
    std::cout << "\nReport saved to: " << filename << "\n";
}

// ==================== MAIN TEST RUNNER ====================

void run_all_benchmarks(const int iterations, const int random_seed, const uint8_t bins) {
    std::cout << "========================================\n";
    std::cout << "    SIMD Functions Benchmark Suite\n";
    std::cout << "========================================\n";

    // Step 1: Validation
    bool hist_valid = validate_histogram(random_seed);
    bool blend_valid = validate_blend(random_seed);

    if (!hist_valid || !blend_valid) {
        std::cerr << "\nERROR: Validation failed! Aborting benchmarks.\n";
        return;
    }

    std::cout << "\n✓ All validations passed!\n";

    // Step 2: Collect all benchmark results
    std::vector<BenchmarkResult> all_results;

    // Synthetic benchmarks
    auto hist_synth_results = benchmark_histogram_synthetic(iterations, random_seed, bins);
    auto blend_synth_results = benchmark_blend_synthetic(iterations, random_seed);

    all_results.insert(all_results.end(), hist_synth_results.begin(), hist_synth_results.end());
    all_results.insert(all_results.end(), blend_synth_results.begin(), blend_synth_results.end());

    // Real image benchmarks (provide your own image paths)
    std::vector<std::string> test_images = {
        "images/hist_test.jpeg", "images/test_background.png",
    };

    std::vector<std::string> foreground_images = {
        "images/test_foreground.png"
    };

    // Uncomment to test with real images:
    auto hist_real_results = benchmark_real_images(iterations, test_images, foreground_images, bins);
    all_results.insert(all_results.end(), hist_real_results.begin(), hist_real_results.end());

    // Step 3: Generate report
    generate_report(all_results);

    // Step 4: Print summary to console
    std::cout << "\n========================================\n";
    std::cout << "           BENCHMARK SUMMARY\n";
    std::cout << "========================================\n\n";

    std::cout << std::left << std::setw(25) << "Function"
              << std::setw(15) << "Best MP/sec"
              << std::setw(15) << "Best Speedup" << "\n";
    std::cout << std::string(55, '-') << "\n";

    // Find best results for each function type
    std::unordered_map<std::string, double> best_mp_per_sec;
    std::unordered_map<std::string, double> best_speedup;

    for (size_t i = 0; i + 1 < all_results.size(); i += 2) {
        const auto& naive_res = all_results[i];
        const auto& simd_res = all_results[i + 1];

        best_mp_per_sec[naive_res.function_name] = std::max(best_mp_per_sec[naive_res.function_name], naive_res.mp_per_sec);
        best_mp_per_sec[simd_res.function_name] = std::max(best_mp_per_sec[simd_res.function_name], simd_res.mp_per_sec);

        double speedup = naive_res.time_μs / simd_res.time_μs;
        best_speedup[simd_res.function_name] = std::max(best_speedup[simd_res.function_name], speedup);
    }

    for (const auto& [func, mp] : best_mp_per_sec) {
        std::cout << std::left << std::setw(25) << func
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << mp;
        if (best_speedup.contains(func))
            std::cout << best_speedup[func] << "x" << std::endl;
        else
            std::cout << std::setw(15) << "-" << std::endl;
    }

    std::cout << "\n========================================\n";
}