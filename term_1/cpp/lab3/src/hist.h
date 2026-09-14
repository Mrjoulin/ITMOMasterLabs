//
// Created by Matthew Ivanov on 08.12.2025.
//

#ifndef LAB2_HIST_H
#define LAB2_HIST_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<uint32_t> histogram_naive(const cv::Mat& image, uint8_t n_bins);
std::vector<uint32_t> histogram_threaded(const cv::Mat& image, uint8_t n_bins, size_t n_threads = 4, size_t block_size = 1024);
std::vector<uint32_t> histogram_simd(const cv::Mat& image, uint8_t n_bins);

#endif //LAB2_HIST_H