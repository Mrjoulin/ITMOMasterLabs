#ifndef LAB2_BLEND_H
#define LAB2_BLEND_H

#include <opencv2/core.hpp>

cv::Mat blend_naive(const cv::Mat &foreground, const cv::Mat &background);
cv::Mat blend_simd(const cv::Mat& foreground, const cv::Mat& background);
cv::Mat blend_simd_old(const cv::Mat& foreground, const cv::Mat& background);

#endif //LAB2_BLEND_H