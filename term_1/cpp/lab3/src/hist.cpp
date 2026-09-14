#include <eve/eve.hpp>
#include "hist.h"

#include <stdatomic.h>

#include "ThreadPool.h"

using wide_uint8 = eve::wide<uint8_t>;

static constexpr bool is_pow2(uint8_t v) {
    return !v || ( !(v & (v - 1)) );
}


std::vector<uint32_t> histogram_naive(const cv::Mat& image, const uint8_t n_bins) {
    // Verify input is grayscale
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(is_pow2(n_bins));

    if (n_bins == 1) return {static_cast<uint32_t>(image.rows * image.cols)};

    uint8_t bin_size = 256 / n_bins;
    uint8_t shift_size = 0;
    while (bin_size >>= 1) ++shift_size;
    std::vector<uint32_t> histogram(n_bins, 0);

    const int total_pixels = image.rows * image.cols;

    // Iterate through all pixels
    for (int i = 0; i < total_pixels; ++i) {
        const uint8_t bin = image.data[i] >> shift_size;
        ++histogram[bin];
    }

    return histogram;
}

void _hist_partial(
    const uint8_t* start_ptr, const uint8_t* end_prt,
    const uint8_t shift_size, std::atomic<uint32_t>* histogram
) {
    std::vector<uint32_t> local_hist(256 >> shift_size, 0);

    for (; start_ptr != end_prt; ++start_ptr) {
        const uint8_t bin = (*start_ptr) >> shift_size;
        ++local_hist[bin];
    }
    for (size_t i = 0; i < local_hist.size(); ++i) {
        histogram[i].fetch_add(local_hist[i], std::memory_order_relaxed);
    }
}

std::vector<uint32_t> histogram_threaded(
    const cv::Mat& image, const uint8_t n_bins,
    const size_t n_threads, const size_t block_size
) {
    // Verify input is grayscale
    CV_Assert(image.type() == CV_8UC1);
    CV_Assert(is_pow2(n_bins));

    if (n_bins == 1) return {static_cast<uint32_t>(image.rows * image.cols)};

    uint8_t bin_size = 256 / n_bins;
    uint8_t shift_size = 0;
    while (bin_size >>= 1) ++shift_size;

    // init hist
    std::atomic<uint32_t> histogram[n_bins];
    for (size_t i = 0; i < n_bins; ++i) {
        histogram[i].store(0, std::memory_order_relaxed);
    }

    // Calc num blocks
    const size_t total_pixels = image.rows * image.cols;
    const size_t _block_size = std::max(block_size, static_cast<size_t>(16));
    const size_t n_blocks = std::ceil(static_cast<double>(total_pixels) / _block_size);

    ThreadPool pool(n_threads);
    for (size_t i = 0; i < n_blocks; i++) {
        const size_t start_idx = i * _block_size;
        const size_t end_idx = std::min(start_idx + _block_size, total_pixels);

        pool.dispatch_task<void>(
            [image, shift_size, start_idx, end_idx, &histogram]() {
                _hist_partial(image.data + start_idx, image.data + end_idx, shift_size, histogram);
            }
        );
    }

    // Wait for results
    pool.stop();

    // convert to vector
    std::vector<uint32_t> result(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
        result[i] = histogram[i].load(std::memory_order_relaxed);
    }

    return result;
}


std::vector<uint32_t> histogram_simd(const cv::Mat& image, const uint8_t n_bins) {
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