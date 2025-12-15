#include <iostream>
#include <eve/arch/spec.hpp>
#include <eve/arch/cpu/wide.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include "benchmark.h"
#include "blend.h"
#include "hist.h"

using Clock = std::chrono::high_resolution_clock;

int manual_test() {

    const std::string bg_image_path = cv::samples::findFile("images/test_background.png");
    const std::string fg_image_path = cv::samples::findFile("images/test_foreground.png");
    cv::Mat bg = cv::imread(bg_image_path, cv::IMREAD_UNCHANGED);
    cv::Mat fg = cv::imread(fg_image_path, cv::IMREAD_UNCHANGED);

    // Resize images to one format
    const int target_width = 1200;
    const int target_height = 900;
    cv::Mat resized_bg;
    cv::Mat resized_fg;
    cv::resize(bg, resized_bg, cv::Size(target_width, target_height));
    cv::resize(fg, resized_fg, cv::Size(target_width, target_height));

    std::cout << "BG channels:" << resized_bg.channels() << std::endl;
    std::cout << "FG channels:" << resized_fg.channels() << std::endl;

    auto start = Clock::now();
    cv::Mat blended = blend_naive(resized_fg, resized_bg);

    auto end = Clock::now();
    double time_μs = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "Native time (μs): " << time_μs << std::endl;

    // cv::imshow("Display window", blended);
    // cv::waitKey(0); // Wait for a keystroke in the window
    cv::imwrite("results/blended_native.png", blended);

    start = Clock::now();
    cv::Mat blended_simd = blend_simd(resized_fg, resized_bg);
    end = Clock::now();
    time_μs = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "SIMD time (μs): " << time_μs << std::endl;

    // cv::imshow("Display window", blended_simd);
    // cv::waitKey(0); // Wait for a keystroke in the window
    cv::imwrite("results/blended_simd.png", blended_simd);

    std::cout << "Binarize images" << std::endl;

    const std::string hist_image_path = cv::samples::findFile("images/hist_test.jpeg");
    cv::Mat hist_test = cv::imread(hist_image_path, cv::IMREAD_GRAYSCALE);

    uint8_t n_bins = 16;

    start = Clock::now();
    auto hist_naive = histogram_naive(hist_test, n_bins);
    end = Clock::now();
    time_μs = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Native time (μs): " << time_μs << std::endl;

    std::cout << "Naive: " << std::endl;
    for (uint32_t i: hist_naive)
        std::cout << i << ' ';
    std::cout << std::endl << std::endl;

    start = Clock::now();
    auto hist_simd = histogram_simd(hist_test, n_bins);
    end = Clock::now();
    time_μs = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "SIMD time (μs): " << time_μs << std::endl;

    std::cout << "SIMD: " << std::endl;
    for (uint32_t i: hist_simd)
        std::cout << i << ' ';
    std::cout << std::endl;

    return 0;
}

int main() {
    constexpr int iterations = 20;
    constexpr int random_seed = 16;
    constexpr uint8_t bins = 16;

    run_all_benchmarks(iterations, random_seed, bins);
    // manual_test();
}
