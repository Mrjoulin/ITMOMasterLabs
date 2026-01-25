#include "benchmark.h"
#include "blend.h"
#include "hist.h"
#include <chrono>
#include <fstream>
#include <algorithm>
#include <sstream>

// ==================== VALIDATION FUNCTIONS ====================

cv::Mat create_synthetic_hist_image(const cv::Size size, std::mt19937& rng) {
    cv::Mat image(size, CV_8UC1);
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

    std::vector<std::pair<cv::Size, uint8_t>> test_cases = {
        {cv::Size(640, 480), 32},
        {cv::Size(1280, 720), 64},
        {cv::Size(1920, 1080), 32},
        {cv::Size(100, 100), 16},
        {cv::Size(7, 13), 8}
    };

    bool all_passed = true;
    std::mt19937 rng(random_seed);

    for (const auto& [size, bins] : test_cases) {
        cv::Mat image = create_synthetic_hist_image(size, rng);

        auto hist_naive = histogram_naive(image, bins);
        auto hist_simd = histogram_simd(image, bins);
        auto hist_threaded = histogram_threaded(image, bins, 4, 2<<10);

        bool passed = true;

        // Compare all three implementations
        std::vector<std::pair<std::string, std::vector<uint32_t>>> implementations = {
            {"simd", hist_simd},
            {"threaded", hist_threaded}
        };

        for (const auto& [name, hist] : implementations) {
            if (hist_naive.size() != hist.size()) {
                std::cout << "FAIL: Size mismatch for " << size << " bins=" << int(bins)
                         << " (naive=" << hist_naive.size() << ", " << name << "=" << hist.size() << ")\n";
                passed = false;
            } else {
                for (size_t i = 0; i < hist_naive.size(); ++i) {
                    if (hist_naive[i] != hist[i]) {
                        std::cout << "FAIL: Bin " << i << " mismatch for " << size
                                 << " bins=" << int(bins) << " (naive=" << hist_naive[i]
                                 << ", " << name << "=" << hist[i] << ")\n";
                        passed = false;
                        break;
                    }
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
        cv::Size(101, 103)
    };

    bool all_passed = true;
    std::mt19937 rng(random_seed);

    for (const auto& size : test_sizes) {
        auto [foreground, background] = create_synthetic_blend_images(size, rng);

        cv::Mat result_naive = blend_naive(foreground, background);
        cv::Mat result_simd = blend_simd(foreground, background);
        cv::Mat result_threaded = blend_threaded(foreground, background, 4, 2<<10);

        bool passed = true;

        // Compare simd and threaded against naive
        std::vector<std::pair<std::string, cv::Mat>> implementations = {
            {"simd", result_simd},
            {"threaded", result_threaded}
        };

        for (const auto& [name, result] : implementations) {
            if (result_naive.size() != result.size() || result_naive.type() != result.type()) {
                std::cout << "FAIL: Size/type mismatch for " << size << " (" << name << ")\n";
                passed = false;
            } else {
                cv::Mat diff;
                cv::absdiff(result_naive, result, diff);
                double max_diff;
                cv::minMaxLoc(diff, nullptr, &max_diff);

                if (max_diff > 1) {
                    std::cout << "FAIL: Pixel mismatch for " << size
                             << " (" << name << ") max_diff=" << max_diff << "\n";
                    passed = false;
                }
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

    template<typename Func>
    static BenchmarkResult find_best_threaded_params(
            const std::string& name,
            const cv::Mat& image,
            Func func_factory,  // Takes (threads, block_size) and returns a callable
            const int iterations,
            const std::vector<size_t>& thread_counts = {2, 4, 6, 8, 10, 12, 14, 16},
            const std::vector<size_t>& block_sizes = {256, 512, 1024, 2048, 2<<12, 2<<13, 2<<14, 2<<15}
    ) {
        double best_time = std::numeric_limits<double>::max();
        size_t best_threads = 0;
        size_t best_block_size = 0;

        // std::cout << "  Testing " << name << " with different parameters:\n";

        for (size_t threads : thread_counts) {
            for (size_t block_size : block_sizes) {
                auto func = func_factory(threads, block_size);
                auto result = run_benchmark(
                    name + "_param_test", image, func, std::max(1, iterations / 10)
                );

                // std::cout << "    Threads=" << threads << ", Block=" << block_size
                //          << " -> " << std::fixed << std::setprecision(2) << result.time_μs << "μs\n";

                if (result.time_μs < best_time) {
                    best_time = result.time_μs;
                    best_threads = threads;
                    best_block_size = block_size;
                }
            }
        }

        // Run final benchmark with best parameters
        auto best_func = func_factory(best_threads, best_block_size);
        BenchmarkResult best_result = run_benchmark(name, image, best_func, iterations);

        std::stringstream params;
        params << "t=" << best_threads << ",b=" << best_block_size;
        best_result.parameters = params.str();
        best_result.function_name = name;

        // std::cout << "  Best: " << best_result.function_name
        //          << " -> " << std::fixed << std::setprecision(2) << best_result.time_μs << "μs\n";

        return best_result;
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

        // Naive implementation
        auto naive_result = Benchmark::run_benchmark(
            "histogram_naive", image, [&]() { histogram_naive(image, bins); }, iterations);
        results.push_back(naive_result);

        // SIMD implementation
        auto simd_result = Benchmark::run_benchmark(
            "histogram_simd", image, [&]() { histogram_simd(image, bins); }, iterations);
        results.push_back(simd_result);

        // Threaded implementation with parameter search
        auto threaded_result = Benchmark::find_best_threaded_params(
            "histogram_threaded", image,
            [&](size_t threads, size_t block_size) {
                return [&, threads, block_size]() {
                    histogram_threaded(image, bins, threads, block_size);
                };
            },
            iterations
        );
        results.push_back(threaded_result);

        double speedup_simd = naive_result.time_μs / simd_result.time_μs;
        double speedup_threaded = naive_result.time_μs / threaded_result.time_μs;

        std::cout << "  Speedup SIMD: " << std::fixed << std::setprecision(2) << speedup_simd << "x\n";
        std::cout << "  Speedup Threaded: " << std::fixed << std::setprecision(2) << speedup_threaded << "x\n";

        if (speedup_threaded > speedup_simd) {
            std::cout << "  Threaded is " << std::fixed << std::setprecision(2)
                     << (speedup_threaded / speedup_simd) << "x faster than SIMD\n";
        } else if (speedup_simd > speedup_threaded) {
            std::cout << "  SIMD is " << std::fixed << std::setprecision(2)
                     << (speedup_simd / speedup_threaded) << "x faster than Threaded\n";
        }
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

        // Naive implementation
        auto naive_result = Benchmark::run_benchmark(
            "blend_naive", foreground,
            [&]() { blend_naive(foreground, background); },
            iterations
        );
        results.push_back(naive_result);

        // SIMD implementation
        auto simd_result = Benchmark::run_benchmark(
            "blend_simd", foreground,
            [&]() { blend_simd(foreground, background); },
            iterations
        );
        results.push_back(simd_result);

        // Threaded implementation with parameter search
        auto threaded_result = Benchmark::find_best_threaded_params(
            "blend_threaded", foreground,
            [&](size_t threads, size_t block_size) {
                return [&, threads, block_size]() {
                    blend_threaded(foreground, background, threads, block_size);
                };
            },
            iterations
        );
        results.push_back(threaded_result);

        double speedup_simd = naive_result.time_μs / simd_result.time_μs;
        double speedup_threaded = naive_result.time_μs / threaded_result.time_μs;

        std::cout << "  Speedup SIMD: " << std::fixed << std::setprecision(2) << speedup_simd << "x\n";
        std::cout << "  Speedup Threaded: " << std::fixed << std::setprecision(2) << speedup_threaded << "x\n";

        if (speedup_threaded > speedup_simd) {
            std::cout << "  Threaded is " << std::fixed << std::setprecision(2)
                     << (speedup_threaded / speedup_simd) << "x faster than SIMD\n";
        } else if (speedup_simd > speedup_threaded) {
            std::cout << "  SIMD is " << std::fixed << std::setprecision(2)
                     << (speedup_simd / speedup_threaded) << "x faster than Threaded\n";
        }
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

    // Blend benchmarks
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

            std::cout << "Testing blend " << foreground.cols << "x" << foreground.rows
                     << " (" << foreground_paths[i] << ")...\n";

            // Naive implementation
            auto naive_result = Benchmark::run_benchmark(
                "blend_naive", foreground,
                [&]() { blend_naive(foreground, background); },
                iterations
            );
            results.push_back(naive_result);

            // SIMD implementation
            auto simd_result = Benchmark::run_benchmark(
                "blend_simd", foreground,
                [&]() { blend_simd(foreground, background); },
                iterations
            );
            results.push_back(simd_result);

            // Threaded implementation with parameter search
            auto threaded_result = Benchmark::find_best_threaded_params(
                "blend_threaded", foreground,
                [&](size_t threads, size_t block_size) {
                    return [&, threads, block_size]() {
                        blend_threaded(foreground, background, threads, block_size);
                    };
                },
                iterations
            );
            results.push_back(threaded_result);

            double speedup_simd = naive_result.time_μs / simd_result.time_μs;
            double speedup_threaded = naive_result.time_μs / threaded_result.time_μs;

            std::cout << "  Speedup SIMD: " << std::fixed << std::setprecision(2) << speedup_simd << "x\n";
            std::cout << "  Speedup Threaded: " << std::fixed << std::setprecision(2) << speedup_threaded << "x\n";
        }
    }

    // Histogram benchmarks
    std::cout << "\n=== Benchmarking Histogram on Real Images ===\n";

    for (const auto& path : image_paths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << "\n";
            continue;
        }

        std::cout << "Testing histogram " << image.cols << "x" << image.rows
                 << " (" << path << ")...\n";

        // Naive implementation
        auto naive_result = Benchmark::run_benchmark(
            "histogram_naive", image,
            [&]() { histogram_naive(image, n_bins); },
            iterations
        );
        results.push_back(naive_result);

        // SIMD implementation
        auto simd_result = Benchmark::run_benchmark(
            "histogram_simd", image,
            [&]() { histogram_simd(image, n_bins); },
            iterations
        );
        results.push_back(simd_result);

        // Threaded implementation with parameter search
        auto threaded_result = Benchmark::find_best_threaded_params(
            "histogram_threaded", image,
            [&](size_t threads, size_t block_size) {
                return [&, threads, block_size]() {
                    histogram_threaded(image, n_bins, threads, block_size);
                };
            },
            iterations
        );
        results.push_back(threaded_result);

        double speedup_simd = naive_result.time_μs / simd_result.time_μs;
        double speedup_threaded = naive_result.time_μs / threaded_result.time_μs;

        std::cout << "  Speedup SIMD: " << std::fixed << std::setprecision(2) << speedup_simd << "x\n";
        std::cout << "  Speedup Threaded: " << std::fixed << std::setprecision(2) << speedup_threaded << "x\n";
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

    report << "=== PERFORMANCE BENCHMARK REPORT ===\n\n";
    report << "Generated: " << __DATE__ << " " << __TIME__ << "\n\n";

    report << std::left << std::setw(30) << "Function"
           << std::setw(15) << "Image Size"
           << std::setw(15) << "Time (μs)"
           << std::setw(15) << "MP/sec"
           << std::setw(15) << "MB/sec"
           << std::setw(20) << "Parameters"
           << std::setw(10) << "Iterations" << "\n";

    report << std::string(120, '-') << "\n";

    for (const auto& result : results) {
        report << std::left << std::setw(30) << result.function_name
               << std::setw(15) << result.image_info
               << std::fixed << std::setprecision(3)
               << std::setw(15) << result.time_μs
               << std::setw(15) << result.mp_per_sec
               << std::setw(15) << result.mb_per_sec
               << std::setw(20) << result.parameters
               << std::setw(10) << result.iterations << "\n";
    }

    // Calculate average speedups for each algorithm type
    struct SpeedupStats {
        double naive_time_sum = 0;
        double simd_time_sum = 0;
        double threaded_time_sum = 0;
        int count = 0;
    };

    std::unordered_map<std::string, SpeedupStats> stats_map; // "blend" or "histogram"

    // Group results by algorithm and image size
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<BenchmarkResult>>> grouped_results;

    for (const auto& result : results) {
        std::string algo_type;

        if (result.function_name.find("blend") != std::string::npos) {
            algo_type = "blend";
        } else if (result.function_name.find("histogram") != std::string::npos) {
            algo_type = "histogram";
        } else {
            continue;
        }

        grouped_results[algo_type][result.image_info].push_back(result);
    }

    // Calculate speedups
    report << "\n=== SPEEDUP ANALYSIS ===\n\n";

    for (const auto& [algo, size_groups] : grouped_results) {
        report << algo << " Algorithm:\n";
        report << std::left << std::setw(15) << "Image Size"
               << std::setw(15) << "SIMD Speedup"
               << std::setw(15) << "Threaded Speedup"
               << std::setw(25) << "Best Parameters" << "\n";
        report << std::string(70, '-') << "\n";

        double simd_speedup_sum = 0;
        double threaded_speedup_sum = 0;
        int valid_comparisons = 0;

        for (const auto& [size, impls] : size_groups) {
            double naive_time = 0;
            double simd_time = 0;
            double threaded_time = 0;
            std::string best_params;

            for (const auto& result : impls) {
                if (result.function_name.find("naive") != std::string::npos) {
                    naive_time = result.time_μs;
                } else if (result.function_name.find("simd") != std::string::npos &&
                          result.function_name.find("param_test") == std::string::npos) {
                    simd_time = result.time_μs;
                } else if (result.function_name.find("threaded") != std::string::npos &&
                          result.function_name.find("param_test") == std::string::npos) {
                    threaded_time = result.time_μs;
                    best_params = result.parameters;
                }
            }

            if (naive_time > 0 && simd_time > 0 && threaded_time > 0) {
                double simd_speedup = naive_time / simd_time;
                double threaded_speedup = naive_time / threaded_time;

                report << std::left << std::setw(15) << size
                       << std::fixed << std::setprecision(2)
                       << std::setw(15) << std::to_string(simd_speedup) + "x"
                       << std::setw(15) << std::to_string(threaded_speedup) + "x"
                       << std::setw(25) << best_params << "\n";

                simd_speedup_sum += simd_speedup;
                threaded_speedup_sum += threaded_speedup;
                valid_comparisons++;
            }
        }

        if (valid_comparisons > 0) {
            report << std::string(70, '-') << "\n";
            report << std::left << std::setw(15) << "Average"
                   << std::fixed << std::setprecision(2)
                   << std::setw(15) << (simd_speedup_sum / valid_comparisons) << "x"
                   << std::setw(15) << (threaded_speedup_sum / valid_comparisons) << "x\n\n";
        }
    }

    report.close();
    std::cout << "\nReport saved to: " << filename << "\n";
}

// ==================== MAIN TEST RUNNER ====================

void run_all_benchmarks(const int iterations, const int random_seed, const uint8_t bins) {
    std::cout << "========================================\n";
    std::cout << "    Performance Benchmark Suite\n";
    std::cout << "    (Naive vs SIMD vs Threaded)\n";
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

    // Real image benchmarks
    std::vector<std::string> test_images = {
        "images/hist_test.jpeg",
        "images/test_background.png",
    };

    std::vector<std::string> foreground_images = {
        "images/test_foreground.png"
    };

    auto real_results = benchmark_real_images(iterations, test_images, foreground_images, bins);
    all_results.insert(all_results.end(), real_results.begin(), real_results.end());

    // Step 3: Generate report
    generate_report(all_results);

    // Step 4: Print summary to console
    std::cout << "\n========================================\n";
    std::cout << "           BENCHMARK SUMMARY\n";
    std::cout << "========================================\n\n";

    // Find best results for each implementation type
    struct SummaryEntry {
        std::string algorithm;
        std::string best_impl;
        double best_speedup;
        double best_mp_per_sec;
        std::string best_params;
    };

    std::vector<SummaryEntry> summary;

    // Group by algorithm and find best implementation
    std::unordered_map<std::string, std::vector<BenchmarkResult>> algo_groups;
    for (const auto& result : all_results) {
        if (result.function_name.find("blend") != std::string::npos) {
            algo_groups["blend"].push_back(result);
        } else if (result.function_name.find("histogram") != std::string::npos) {
            algo_groups["histogram"].push_back(result);
        }
    }

    std::cout << std::left << std::setw(20) << "Algorithm"
              << std::setw(20) << "Best Implementation"
              << std::setw(15) << "Best Speedup"
              << std::setw(15) << "Best MP/sec"
              << std::setw(20) << "Parameters" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& [algo, results] : algo_groups) {
        // Find naive baseline for speedup calculation
        double naive_mp_per_sec = 0;
        for (const auto& result : results) {
            if (result.function_name.find("naive") != std::string::npos) {
                naive_mp_per_sec = result.mp_per_sec;
                break;
            }
        }

        // Find best implementation
        const BenchmarkResult* best_result = nullptr;
        double best_speedup = 0;

        for (const auto& result : results) {
            if (result.function_name.find("naive") == std::string::npos &&
                result.function_name.find("param_test") == std::string::npos) {
                double speedup = result.mp_per_sec / naive_mp_per_sec;
                if (speedup > best_speedup) {
                    best_speedup = speedup;
                    best_result = &result;
                }
            }
        }

        if (best_result) {
            std::cout << std::left << std::setw(20) << algo
                      << std::setw(20) << best_result->function_name.substr(0, 20)
                      << std::fixed << std::setprecision(2)
                      << std::setw(15) << best_speedup << "x"
                      << std::setw(15) << best_result->mp_per_sec
                      << std::setw(20) << best_result->parameters << "\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "           RECOMMENDATIONS\n";
    std::cout << "========================================\n\n";

    // Generate recommendations based on results
    for (const auto& [algo, results] : algo_groups) {
        std::cout << algo << ":\n";

        // Count wins for each implementation
        int simd_wins = 0;
        int threaded_wins = 0;
        int total_comparisons = 0;

        std::unordered_map<std::string, std::pair<double, double>> size_comparisons;

        for (const auto& result : results) {
            std::string size = result.image_info;
            if (result.function_name.find("simd") != std::string::npos &&
                result.function_name.find("param_test") == std::string::npos) {
                size_comparisons[size].first = result.mp_per_sec;
            } else if (result.function_name.find("threaded") != std::string::npos &&
                      result.function_name.find("param_test") == std::string::npos) {
                size_comparisons[size].second = result.mp_per_sec;
            }
        }

        for (const auto& [size, speeds] : size_comparisons) {
            if (speeds.first > 0 && speeds.second > 0) {
                total_comparisons++;
                if (speeds.first > speeds.second) {
                    simd_wins++;
                } else {
                    threaded_wins++;
                }
            }
        }

        std::cout << "  SIMD won " << simd_wins << "/" << total_comparisons << " comparisons\n";
        std::cout << "  Threaded won " << threaded_wins << "/" << total_comparisons << " comparisons\n";

        if (simd_wins > threaded_wins) {
            std::cout << "  → Recommendation: Use SIMD implementation\n";
        } else if (threaded_wins > simd_wins) {
            std::cout << "  → Recommendation: Use Threaded implementation\n";
        } else {
            std::cout << "  → Recommendation: Both perform similarly, choose based on constraints\n";
        }
        std::cout << "\n";
    }

    std::cout << "========================================\n";
}