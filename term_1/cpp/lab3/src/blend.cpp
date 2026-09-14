#include <opencv2/core.hpp>
#include <eve/wide.hpp>
#include <eve/module/core.hpp>

#include "blend.h"
#include "ThreadPool.h"

using wide_uint16 = eve::wide<uint16_t>;
using wide_uint32 = eve::wide<uint32_t>;


cv::Mat blend_naive(const cv::Mat &foreground, const cv::Mat &background) {
    CV_Assert(foreground.type() == CV_8UC4 && background.type() == CV_8UC4);
    CV_Assert(foreground.rows == background.rows && foreground.cols == background.cols);

    constexpr uint32_t pixel_max = 255;
    constexpr uint32_t pixel_max_sq = pixel_max * pixel_max;

    cv::Mat result = cv::Mat::zeros(background.rows, background.cols, background.type());

    for (int i = 0; i < background.rows; i++) {
        for (int j = 0; j < background.cols; j++) {
            const auto& fg_pixel = foreground.at<cv::Vec4b>(i, j);
            const auto& bg_pixel = background.at<cv::Vec4b>(i, j);
            auto & res_pixel = result.at<cv::Vec4b>(i, j);

            const uint32_t fg_alpha = fg_pixel[3];
            const uint32_t fg_inv_alpha = pixel_max - fg_alpha;
            const uint32_t bg_comb_alpha = fg_inv_alpha * bg_pixel[3] / pixel_max;

            // Blend channels
            res_pixel[0] = static_cast<uint8_t>(
                (static_cast<uint32_t>(fg_pixel[0]) * fg_alpha + static_cast<uint32_t>(bg_pixel[0]) * bg_comb_alpha) / pixel_max
            );
            res_pixel[1] = static_cast<uint8_t>(
                (static_cast<uint32_t>(fg_pixel[1]) * fg_alpha + static_cast<uint32_t>(bg_pixel[1]) * bg_comb_alpha) / pixel_max
            );
            res_pixel[2] = static_cast<uint8_t>(
                (static_cast<uint32_t>(fg_pixel[2]) * fg_alpha + static_cast<uint32_t>(bg_pixel[2]) * bg_comb_alpha) / pixel_max
            );
            res_pixel[3] = static_cast<uint8_t>(
                (pixel_max_sq - fg_inv_alpha * (pixel_max - bg_pixel[3])) / pixel_max
            );
        }
    }

    return result;
}


void _blend_partial(
    const cv::Mat& foreground, const cv::Mat& background, std::uint8_t* result,
    const size_t start_idx, const size_t end_idx
) {
    CV_Assert(foreground.type() == CV_8UC4 && background.type() == CV_8UC4);
    CV_Assert(foreground.rows == background.rows && foreground.cols == background.cols);

    constexpr uint32_t pixel_max = 255;
    constexpr uint32_t pixel_max_sq = pixel_max * pixel_max;

    constexpr int channels = 4;
    const int cols = background.cols;

    // Process remaining pixels (scalar fallback)
    for (size_t i = start_idx; i < end_idx; i += channels) {
        // Calculate pixel index
        const int pixel_idx = static_cast<int>(i / channels);
        const int row = pixel_idx / cols;
        const int col = pixel_idx % cols;

        const auto& fg_pixel = foreground.at<cv::Vec4b>(row, col);
        const auto& bg_pixel = background.at<cv::Vec4b>(row, col);
        const auto res_pixel = result + i;

        const uint32_t fg_alpha = fg_pixel[3];
        const uint32_t fg_inv_alpha = pixel_max - fg_alpha;
        const uint32_t bg_comb_alpha = fg_inv_alpha * bg_pixel[3] / pixel_max;

        res_pixel[0] = static_cast<uint8_t>(
            (static_cast<uint32_t>(fg_pixel[0]) * fg_alpha + static_cast<uint32_t>(bg_pixel[0]) * bg_comb_alpha) / pixel_max
        );
        res_pixel[1] = static_cast<uint8_t>(
            (static_cast<uint32_t>(fg_pixel[1]) * fg_alpha + static_cast<uint32_t>(bg_pixel[1]) * bg_comb_alpha) / pixel_max
        );
        res_pixel[2] = static_cast<uint8_t>(
            (static_cast<uint32_t>(fg_pixel[2]) * fg_alpha + static_cast<uint32_t>(bg_pixel[2]) * bg_comb_alpha) / pixel_max
        );
        res_pixel[3] = static_cast<uint8_t>(
            (pixel_max_sq - fg_inv_alpha * (pixel_max - bg_pixel[3])) / pixel_max
        );
    }
}


cv::Mat blend_threaded(
    const cv::Mat &foreground, const cv::Mat &background,
    const size_t n_threads, const size_t block_size
) {
    CV_Assert(foreground.type() == CV_8UC4 && background.type() == CV_8UC4);
    CV_Assert(foreground.rows == background.rows && foreground.cols == background.cols);

    constexpr int channels = 4;
    const int rows = background.rows;
    const int cols = background.cols;
    const size_t full_size = rows * cols;
    const size_t full_size_bytes = full_size * channels;
    const size_t _block_size = std::max(block_size, static_cast<size_t>(16));
    const size_t n_blocks = std::ceil(static_cast<double>(full_size) / _block_size);

    cv::Mat result = cv::Mat::zeros(background.rows, background.cols, background.type());

    ThreadPool pool(n_threads);

    for (size_t i = 0; i < n_blocks; i++) {
        const size_t start_idx = i * _block_size * channels;
        const size_t end_idx = std::min(start_idx + _block_size * channels, full_size_bytes);

        pool.dispatch_task<void>(
            [foreground, background, result, start_idx, end_idx] () {
                _blend_partial(foreground, background, result.data, start_idx, end_idx);
            }
        );
    }

    // Wait for results
    pool.stop();
    return result;
}



cv::Mat blend_simd(const cv::Mat& foreground, const cv::Mat& background) {
    CV_Assert(foreground.type() == CV_8UC4 && background.type() == CV_8UC4);
    CV_Assert(foreground.rows == background.rows && foreground.cols == background.cols);

    constexpr uint32_t pixel_max = 255;
    constexpr uint32_t pixel_max_sq = pixel_max * pixel_max;

    constexpr int channels = 4;
    const int rows = background.rows;
    const int cols = background.cols;

    constexpr size_t simd_width = wide_uint32::size(); // 8
    constexpr size_t simd_bytes = simd_width * channels;
    const size_t full_size = rows * cols * channels;
    const size_t simd_size = full_size - (full_size % simd_bytes);

    cv::Mat result = cv::Mat::zeros(rows, cols, background.type());

    // Create wide constants
    const auto wide_pixel_max = wide_uint32(pixel_max);
    const auto wide_pixel_max_sq = wide_uint32(pixel_max_sq);

    struct SIMD_RGBA {
        wide_uint32 b, g, r, a;
    };

    // Main SIMD processing loop
    for (size_t i = 0; i < simd_size; i += simd_bytes) {
        // Load 4 RGBA pixels (16 bytes)
        auto fg_ptr = reinterpret_cast<uint32_t*>(foreground.data + i);
        auto bg_ptr =  reinterpret_cast<uint32_t*>(background.data + i);

        wide_uint32 fg_pix = eve::load(fg_ptr, eve::as<wide_uint32>{});
        wide_uint32 bg_pix = eve::load(bg_ptr, eve::as<wide_uint32>{});

        // Split each channel to own register
        SIMD_RGBA fg, bg;

        fg.b = eve::bit_and(fg_pix, wide_pixel_max);
        fg.g = eve::bit_and(eve::bit_shr(fg_pix, 8), wide_pixel_max);
        fg.r = eve::bit_and(eve::bit_shr(fg_pix, 16), wide_pixel_max);
        fg.a = eve::bit_shr(fg_pix, 24);

        bg.b = eve::bit_and(bg_pix, wide_pixel_max);
        bg.g = eve::bit_and(eve::bit_shr(bg_pix, 8), wide_pixel_max);
        bg.r = eve::bit_and(eve::bit_shr(bg_pix, 16), wide_pixel_max);
        bg.a = eve::bit_shr(bg_pix, 24);

        // Calculate inverse alpha
        const auto bg_comb_alpha = (wide_pixel_max - fg.a) * bg.a / wide_pixel_max;

        // Blend channels
        auto res_blue = (fg.b * fg.a + bg.b * bg_comb_alpha) / wide_pixel_max;
        auto res_green = (fg.g * fg.a + bg.g * bg_comb_alpha) / wide_pixel_max;
        auto res_red = (fg.r * fg.a + bg.r * bg_comb_alpha) / wide_pixel_max;
        auto res_alpha = (wide_pixel_max_sq - (wide_pixel_max - fg.a) * (wide_pixel_max - bg.a)) / pixel_max;

        // Combine all
        auto result_pixels = eve::bit_shl(res_alpha, 24) + eve::bit_shl(res_red, 16) + eve::bit_shl(res_green, 8) + res_blue;

        // Store back to BGRA format
        auto result_ptr = reinterpret_cast<uint32_t*>(result.data + i);
        eve::store(result_pixels, result_ptr);
    }

    // Process remaining pixels (scalar fallback)
    _blend_partial(foreground, background, result.data, simd_size, full_size);

    return result;
}
