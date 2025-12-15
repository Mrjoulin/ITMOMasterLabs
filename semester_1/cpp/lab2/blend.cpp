#include <opencv2/core.hpp>
#include <eve/wide.hpp>
#include <eve/module/core.hpp>

#include "blend.h"

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
    for (size_t i = simd_size; i < full_size; i += channels) {
        // Calculate pixel index
        const int pixel_idx = static_cast<int>(i / channels);
        const int row = pixel_idx / cols;
        const int col = pixel_idx % cols;

        const auto& fg_pixel = foreground.at<cv::Vec4b>(row, col);
        const auto& bg_pixel = background.at<cv::Vec4b>(row, col);
        auto& res_pixel = result.at<cv::Vec4b>(row, col);

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

    return result;
}


cv::Mat blend_simd_old(const cv::Mat& foreground, const cv::Mat& background) {
    CV_Assert(foreground.type() == CV_8UC4 && background.type() == CV_8UC4);
    CV_Assert(foreground.rows == background.rows && foreground.cols == background.cols);

    constexpr uint16_t pixel_max = 255;
    constexpr uint16_t pixel_max_sq = pixel_max * pixel_max;

    constexpr int channels = 4;
    const int rows = background.rows;
    const int cols = background.cols;

    constexpr size_t simd_width = wide_uint16::size(); // 8
    constexpr size_t simd_bytes = simd_width * 4;
    const size_t full_size = rows * cols * channels;
    const size_t simd_size = full_size - (full_size % simd_bytes);

    cv::Mat result = cv::Mat::zeros(rows, cols, background.type());

    // Create wide constants
    const auto wide_pixel_max = wide_uint16(pixel_max);
    const auto wide_pixel_max_sq = wide_uint16(pixel_max_sq);

    struct SIMD_RGBA {
        wide_uint16 b, g, r, a;
    };

    // Main SIMD processing loop
    for (size_t i = 0; i < simd_size; i += simd_bytes) {
        // Load 4 RGBA pixels (16 bytes)

        // Load each channel
        auto fg_ptr = foreground.data + i;
        auto bg_ptr = background.data + i;

        SIMD_RGBA fg, bg;

        fg.b = {[fg_ptr](auto j, auto ) { return *(fg_ptr + j*4); }};
        fg.g = {[fg_ptr](auto j, auto ) { return *(fg_ptr + j*4 + 1); }};
        fg.r = {[fg_ptr](auto j, auto ) { return *(fg_ptr + j*4 + 2); }};
        fg.a = {[fg_ptr](auto j, auto ) { return *(fg_ptr + j*4 + 3); }};

        bg.b = {[bg_ptr](auto j, auto ) { return *(bg_ptr + j*4); }};
        bg.g = {[bg_ptr](auto j, auto ) { return *(bg_ptr + j*4 + 1); }};
        bg.r = {[bg_ptr](auto j, auto ) { return *(bg_ptr + j*4 + 2); }};
        bg.a = {[bg_ptr](auto j, auto ) { return *(bg_ptr + j*4 + 3); }};

        // Calculate inverse alpha
        auto fg_inv_alpha = wide_pixel_max - fg.a;

        // Blend blue channel
        auto blue_term1 = (fg.b * fg.a) / wide_pixel_max;
        auto blue_term2 = ((bg.b * fg_inv_alpha) / wide_pixel_max) * bg.a / wide_pixel_max;
        auto res_blue = blue_term1 + blue_term2;

        // Blend green channel
        auto green_term1 = (fg.g * fg.a) / wide_pixel_max;
        auto green_term2 = ((bg.g * fg_inv_alpha) / wide_pixel_max) * bg.a / wide_pixel_max;
        auto res_green = green_term1 + green_term2;

        // Blend red channel
        auto red_term1 = (fg.r * fg.a) / wide_pixel_max;
        auto red_term2 = ((bg.r * fg_inv_alpha) / wide_pixel_max) * bg.a / wide_pixel_max;
        auto res_red = red_term1 + red_term2;

        // Blend alpha channel
        auto alpha_temp = fg_inv_alpha * (wide_pixel_max - bg.a);
        auto res_alpha = (wide_pixel_max_sq - alpha_temp) / wide_pixel_max;

        // Combine blue and green, red and alpha
        auto res_blue_green = eve::convert(eve::bit_shl(res_green, 8) + res_blue, eve::as<uint32_t>{});
        auto res_red_alpha = eve::convert(eve::bit_shl(res_alpha, 8) + res_red, eve::as<uint32_t>{});
        // Combine all
        auto result_pixels = eve::bit_shl(res_red_alpha, 16) + res_blue_green;

        // Store back to BGRA format
        auto result_ptr = reinterpret_cast<uint32_t*>(result.data + i);
        eve::store(result_pixels, result_ptr);
    }

    // Process remaining pixels (scalar fallback)
    for (size_t i = simd_size; i < full_size; i += channels) {
        // Calculate pixel index
        const int pixel_idx = static_cast<int>(i / channels);
        const int row = pixel_idx / cols;
        const int col = pixel_idx % cols;

        const auto& fg_pixel = foreground.at<cv::Vec4b>(row, col);
        const auto& bg_pixel = background.at<cv::Vec4b>(row, col);
        auto& res_pixel = result.at<cv::Vec4b>(row, col);

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

    return result;
}
