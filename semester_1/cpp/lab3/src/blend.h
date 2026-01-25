#ifndef LAB2_BLEND_H
#define LAB2_BLEND_H

#include <opencv2/core.hpp>

cv::Mat blend_naive(const cv::Mat &foreground, const cv::Mat &background);
cv::Mat blend_threaded(const cv::Mat &foreground, const cv::Mat &background, size_t n_threads = 10, size_t block_size = 1024);
cv::Mat blend_simd(const cv::Mat& foreground, const cv::Mat& background);

#endif //LAB2_BLEND_H