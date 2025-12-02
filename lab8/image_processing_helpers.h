#ifndef IMAGE_PROCESSING_HELPERS_H
#define IMAGE_PROCESSING_HELPERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ==================== OPERAȚIUNI PUNCTUALE ====================

/**
 * @brief Negativează o imagine (inversează valorile pixelilor)
 * @param img Pointer către datele imaginii
 * @param w Lățimea imaginii
 * @param h Înălțimea imaginii
 * @return Pointer către noua imagine negativată (trebuie dealocată manual)
 */
unsigned char* negateImage(unsigned char* img, int w, int h);

/**
 * @brief Binarizează o imagine folosind un prag
 * @param img Imaginea de intrare (grayscale)
 * @param threshold Pragul de binarizare (default: 128)
 * @return Imaginea binarizată
 */
cv::Mat binarizeImage(const cv::Mat& img, int threshold = 128);

/**
 * @brief Egalizează histograma unei imagini
 * @param img Imaginea de intrare (grayscale)
 * @return Imaginea cu histograma egalizată
 */
cv::Mat equalizeHistogram(const cv::Mat& img);

/**
 * @brief Calculează histograma unei imagini
 * @param img Imaginea de intrare (grayscale)
 * @return Histograma calculată
 */
cv::Mat calculateHistogram(const cv::Mat& img);

/**
 * @brief Desenează histograma ca imagine
 * @param hist Histograma de desenat
 * @param hist_w Lățimea imaginii histogramei (default: 512)
 * @param hist_h Înălțimea imaginii histogramei (default: 400)
 * @return Imaginea cu histograma desenată
 */
cv::Mat drawHistogram(const cv::Mat& hist, int hist_w = 512, int hist_h = 400);

// ==================== OPERAȚIUNI SPAȚIALE ====================

/**
 * @brief Aplică filtrul Sobel personalizat
 * @param img Pointer către datele imaginii
 * @param w Lățimea imaginii
 * @param h Înălțimea imaginii
 * @return Pointer către noua imagine (trebuie dealocată manual)
 */
unsigned char* sobelImage(unsigned char* img, int w, int h);

/**
 * @brief Aplică Motion Blur pe orizontală
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 5)
 * @return Imaginea cu motion blur aplicat
 */
cv::Mat applyMotionBlur(const cv::Mat& img, int kernelSize = 5);

/**
 * @brief Aplică Gaussian Blur
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 5)
 * @return Imaginea cu gaussian blur aplicat
 */
cv::Mat applyGaussianBlur(const cv::Mat& img, int kernelSize = 5);

/**
 * @brief Aplică filtrul Laplacian
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 3)
 * @return Imaginea cu Laplacian aplicat
 */
cv::Mat applyLaplacian(const cv::Mat& img, int kernelSize = 3);

/**
 * @brief Aplică filtrul Sobel (ambele direcții)
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 3)
 * @return Imaginea cu Sobel aplicat
 */
cv::Mat applySobel(const cv::Mat& img, int kernelSize = 3);

/**
 * @brief Aplică Mean Blur (media aritmetică)
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 5)
 * @return Imaginea cu mean blur aplicat
 */
cv::Mat applyMeanBlur(const cv::Mat& img, int kernelSize = 5);

/**
 * @brief Detectează muchii pe orizontală (Sobel X)
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 3)
 * @return Imaginea cu muchii orizontale detectate
 */
cv::Mat detectHorizontalEdges(const cv::Mat& img, int kernelSize = 3);

/**
 * @brief Detectează muchii pe verticală (Sobel Y)
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (default: 3)
 * @return Imaginea cu muchii verticale detectate
 */
cv::Mat detectVerticalEdges(const cv::Mat& img, int kernelSize = 3);

/**
 * @brief Combină muchiile orizontale și verticale
 * @param horizontal Imaginea cu muchii orizontale
 * @param vertical Imaginea cu muchii verticale
 * @return Imaginea cu muchii combinate
 */
cv::Mat combineEdges(const cv::Mat& horizontal, const cv::Mat& vertical);

/**
 * @brief Aplică Median Blur
 * @param img Imaginea de intrare
 * @param kernelSize Dimensiunea kernel-ului (trebuie să fie impar, default: 3)
 * @return Imaginea cu median blur aplicat
 */
cv::Mat applyMedianBlur(const cv::Mat& img, int kernelSize = 3);

#endif // IMAGE_PROCESSING_HELPERS_H