#ifndef FREQUENCY_DOMAIN_HELPERS_H
#define FREQUENCY_DOMAIN_HELPERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

// ==================== TRANSFORMATA FOURIER ====================

cv::Mat computeFourierSpectrum(const cv::Mat& src, cv::Mat& complexImage);
cv::Mat computeDFT(const cv::Mat& src);
cv::Mat computeIDFT(const cv::Mat& complexImage);
cv::Mat shiftDFT(const cv::Mat& complexImage);
cv::Mat computeMagnitudeSpectrum(const cv::Mat& complexImage);

// ==================== FILTRE TRECE-JOS ====================

cv::Mat createIdealLowPassFilter(int rows, int cols, float D0);
cv::Mat createButterworthLowPassFilter(int rows, int cols, float D0, int n = 2);
cv::Mat createGaussianLowPassFilter(int rows, int cols, float D0);

// ==================== FILTRE TRECE-SUS ====================

cv::Mat createIdealHighPassFilter(int rows, int cols, float D0);
cv::Mat createButterworthHighPassFilter(int rows, int cols, float D0, int n = 2);
cv::Mat createGaussianHighPassFilter(int rows, int cols, float D0);

// ==================== APLICARE FILTRE ====================

cv::Mat applyFrequencyFilter(const cv::Mat& src, const cv::Mat& filter);
cv::Mat multiplyComplexWithFilter(const cv::Mat& complexImage, const cv::Mat& filter);

// ==================== FUNCȚII UTILITARE ====================

float computeDistance(int u, int v, int centerU, int centerV);
cv::Mat getOptimalDFTSize(const cv::Mat& src);
cv::Mat spatialFilterToFrequencyDomain(const cv::Mat& spatialFilter, int targetRows, int targetCols);

#endif // FREQUENCY_DOMAIN_HELPERS_H
