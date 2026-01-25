#include "src/benchmark.h"
#include "src/blend.h"
#include "src/hist.h"

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

    std::cout << "Naive time (μs): " << time_μs << std::endl;

    // cv::imshow("Display window", blended);
    // cv::waitKey(0); // Wait for a keystroke in the window
    cv::imwrite("results/blended_naive.png", blended);

    start = Clock::now();
    cv::Mat blended_simd = blend_simd(resized_fg, resized_bg);
    end = Clock::now();
    time_μs = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "SIMD time (μs): " << time_μs << std::endl;

    // cv::imshow("Display window", blended_simd);
    // cv::waitKey(0); // Wait for a keystroke in the window
    cv::imwrite("results/blended_simd.png", blended_simd);

    std::vector thr_ns = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};
    std::vector blk_szs = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2<<14, 2<<15};

    double best_time = 1e8;
    std::pair best_params = {0, 0};
    cv::Mat blended_thr;

    for (auto thr_n: thr_ns) {
        for (auto blk_sz: blk_szs) {
            start = Clock::now();
            blended_thr = blend_threaded(resized_fg, resized_bg, thr_n, blk_sz);
            end = Clock::now();
            time_μs = std::chrono::duration<double, std::micro>(end - start).count();
            std::cout << "Threaded time (μs) thr=" << thr_n << ", blk=" << blk_sz << ": " << time_μs << std::endl;
            if (time_μs < best_time) {
                best_time = time_μs;
                best_params = {thr_n, blk_sz};
            }
        }
    }
    std::cout << std::endl << "Best params: " << best_params.first << " thr, " << best_params.second
        << " block_size; Time: " << best_time << std::endl;

    // cv::imshow("Display window", blended_simd);
    // cv::waitKey(0); // Wait for a keystroke in the window
    cv::imwrite("results/blended_thr.png", blended_thr);

    std::cout << "Binarize images" << std::endl;

    const std::string hist_image_path = cv::samples::findFile("images/hist_test.jpeg");
    cv::Mat hist_test = cv::imread(hist_image_path, cv::IMREAD_GRAYSCALE);

    uint8_t n_bins = 16;

    start = Clock::now();
    auto hist_naive = histogram_naive(hist_test, n_bins);
    end = Clock::now();
    time_μs = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Naive time (μs): " << time_μs << std::endl;

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

    std::vector<uint32_t> hist_thr;
    best_time = 1e8;
    best_params = {0, 0};

    for (auto thr_n: thr_ns) {
        for (auto blk_sz: blk_szs) {
            start = Clock::now();
            hist_thr = histogram_threaded(hist_test, n_bins, thr_n, blk_sz);
            end = Clock::now();
            time_μs = std::chrono::duration<double, std::micro>(end - start).count();
            std::cout << "Threaded time (μs) thr=" << thr_n << ", blk=" << blk_sz << ": " << time_μs << std::endl;
            if (time_μs < best_time) {
                best_time = time_μs;
                best_params = {thr_n, blk_sz};
            }
        }
    }
    std::cout << std::endl << "Best params: " << best_params.first << " thr, " << best_params.second
        << " block_size; Time: " << best_time << std::endl;

    std::cout << "Threaded: " << std::endl;
    for (uint32_t i: hist_thr)
        std::cout << i << ' ';
    std::cout << std::endl << std::endl;

    return 0;
}

void test_speed_params_blend() {
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

    std::vector thr_ns = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector blk_szs = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2<<14, 2<<15};

    std::vector<double> times;
    std::vector<std::pair<int, int>> params;

    double best_time = 1e8;
    std::pair best_params = {0, 0};

    for (auto thr_n: thr_ns) {
        for (auto blk_sz: blk_szs) {
            auto start = Clock::now();
            blend_threaded(resized_fg, resized_bg, thr_n, blk_sz);
            auto end = Clock::now();
            double time_μs = std::chrono::duration<double, std::micro>(end - start).count();
            // std::cout << "Threaded time (μs) thr=" << thr_n << ", blk=" << blk_sz << ": " << time_μs << std::endl;
            if (time_μs < best_time) {
                best_time = time_μs;
                best_params = {thr_n, blk_sz};
            }
            times.push_back(time_μs);
            params.push_back({thr_n, blk_sz });
        }
    }
    std::cout << std::endl << "Best params: " << best_params.first << " thr, " << best_params.second
        << " block_size; Time: " << best_time << std::endl;

    std::cout << "Times:"<< std::endl;
    for (double i: times) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    std::cout << "Params:"<< std::endl;
    for (auto p: params) {
        std::cout << p.first << "," << p.second << ' ';
    }
    std::cout << std::endl;
}



void test_speed_params_hist() {
    const std::string bg_image_path = cv::samples::findFile("images/test_background.png");
    cv::Mat bg = cv::imread(bg_image_path, cv::IMREAD_GRAYSCALE);

    std::vector thr_ns = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector blk_szs = {2048, 4096, 8192, 16384, 2<<14, 2<<15};

    std::vector<double> times;
    std::vector<std::pair<int, int>> params;

    double best_time = 1e8;
    std::pair best_params = {0, 0};

    for (auto thr_n: thr_ns) {
        for (auto blk_sz: blk_szs) {
            auto start = Clock::now();
            histogram_threaded(bg, 16, thr_n, blk_sz);
            auto end = Clock::now();
            double time_μs = std::chrono::duration<double, std::micro>(end - start).count();
            // std::cout << "Threaded time (μs) thr=" << thr_n << ", blk=" << blk_sz << ": " << time_μs << std::endl;
            if (time_μs < best_time) {
                best_time = time_μs;
                best_params = {thr_n, blk_sz};
            }
            times.push_back(time_μs);
            params.push_back({thr_n, blk_sz });
        }
    }
    std::cout << std::endl << "Best params: " << best_params.first << " thr, " << best_params.second
        << " block_size; Time: " << best_time << std::endl;

    std::cout << "Times:"<< std::endl;
    for (double i: times) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    std::cout << "Params:"<< std::endl;
    for (auto p: params) {
        std::cout << p.first << "," << p.second << ' ';
    }
    std::cout << std::endl;
}

int main() {
    constexpr int iterations = 20;
    constexpr int random_seed = 16;
    constexpr uint8_t bins = 16;

    run_all_benchmarks(iterations, random_seed, bins);
    // manual_test();
    // test_speed_params_hist();
}
