#include <eve/eve.hpp>
#include "hist.h"

using wide_uint8 = eve::wide<uint8_t>;

static constexpr bool is_pow2(uint8_t v) {
    return !v || ( !(v & (v - 1)) );
}


std::vector<uint32_t> histogram_naive(const cv::Mat& image, uint8_t n_bins) {
    // Verify input is grayscale
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(is_pow2(n_bins));

    if (n_bins == 1) return {static_cast<uint32_t>(image.rows * image.cols)};

    uint8_t bin_size = 256 / n_bins;
    uint8_t shift_size = 0;
    while (bin_size >>= 1) ++shift_size;
    std::vector<uint32_t> histogram(n_bins, 0);

    // Iterate through all pixels
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            const uint8_t pixel_value = image.at<uint8_t>(i, j);
            const uint8_t bin = pixel_value >> shift_size;
            ++histogram[bin];
        }
    }

    return histogram;
}


std::vector<uint32_t> histogram_simd(const cv::Mat& image, uint8_t n_bins) {
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(is_pow2(n_bins));

    if (n_bins == 1) return {static_cast<uint32_t>(image.rows * image.cols)};

    uint8_t bin_size = 256 / n_bins;
    uint8_t shift_size = 0;
    while (bin_size >>= 1) ++shift_size;

    const int rows = image.rows;
    const int cols = image.cols;
    const int total_pixels = rows * cols;

    // Initialize histogram with zeros
    std::vector<uint32_t> histogram(n_bins, 0);
    constexpr size_t simd_width = wide_uint8::size();
    const size_t simd_size = total_pixels - (total_pixels % simd_width);

    for (size_t i = 0; i < simd_size; i += simd_width) {
        wide_uint8 pixel_vector = eve::load(image.data + i, eve::as<wide_uint8>{});
        wide_uint8 pixel_bins = eve::bit_shr(pixel_vector, shift_size);

        for (size_t j = 0; j < simd_width; ++j) {++histogram[pixel_bins.get(j)];}
    }

    // Process remaining pixels (scalar fallback)
    for (size_t i = simd_size; i < total_pixels; ++i) {
        const uint8_t bin = image.data[i] >> shift_size;
        ++histogram[bin];
    }

    return histogram;
}