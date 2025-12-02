#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>

// ==================== HELPER FUNCTIONS ====================

// ==================== OPERAȚIUNI PUNCTUALE ====================

// 1a. Negativare imagine
unsigned char* negateImage(unsigned char* img, int w, int h)
{
    unsigned char* result = new unsigned char[w*h];
    cv::Mat inMat(h, w, CV_8UC1, img);
    cv::Mat negateMat(h, w, CV_8UC1, result);
    cv::bitwise_not(inMat, negateMat);
    return result;
}

// 1a. Binarizare imagine
cv::Mat binarizeImage(const cv::Mat& img, int threshold = 128)
{
    cv::Mat binaryImg;
    cv::threshold(img, binaryImg, threshold, 255, cv::THRESH_BINARY);
    return binaryImg;
}

// 1b. Egalizare histogramă
cv::Mat equalizeHistogram(const cv::Mat& img)
{
    cv::Mat equalizedImg;
    cv::equalizeHist(img, equalizedImg);
    return equalizedImg;
}

// 1b. Calcul histogramă
cv::Mat calculateHistogram(const cv::Mat& img)
{
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    return hist;
}

// 1b. Desenare histogramă
cv::Mat drawHistogram(const cv::Mat& hist, int hist_w = 512, int hist_h = 400)
{
    int histSize = 256;
    int bin_w = cvRound((double) hist_w/histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));

    cv::Mat hist_normalized;
    cv::normalize(hist, hist_normalized, 0, histImage.rows, cv::NORM_MINMAX);

    for(int i = 1; i < histSize; i++) {
        cv::line(histImage,
                 cv::Point(bin_w*(i-1), hist_h - cvRound(hist_normalized.at<float>(i-1))),
                 cv::Point(bin_w*(i), hist_h - cvRound(hist_normalized.at<float>(i))),
                 cv::Scalar(0), 2);
    }

    return histImage;
}

// ==================== OPERAȚIUNI SPAȚIALE ====================

// 2a. Filtru Sobel personalizat
unsigned char* sobelImage(unsigned char* img, int w, int h)
{
    unsigned char* result = new unsigned char[w*h];
    cv::Mat inMat(h, w, CV_8UC1, img);
    cv::Mat outMat;
    cv::Mat kern = (cv::Mat_<char>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::filter2D(inMat, outMat, inMat.depth(), kern);
    cv::Mat absMat(h, w, CV_8UC1, result);
    cv::convertScaleAbs(outMat, absMat);
    return result;
}

// 2a. Motion Blur pe orizontală
cv::Mat applyMotionBlur(const cv::Mat& img, int kernelSize = 5)
{
    cv::Mat motionBlur;
    cv::Mat kernel_motion = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
    for(int i = 0; i < kernelSize; i++) {
        kernel_motion.at<float>(kernelSize/2, i) = 1.0/kernelSize;
    }
    cv::filter2D(img, motionBlur, -1, kernel_motion);
    return motionBlur;
}

// 2a. Gaussian Blur
cv::Mat applyGaussianBlur(const cv::Mat& img, int kernelSize = 5)
{
    cv::Mat gaussianBlur;
    cv::GaussianBlur(img, gaussianBlur, cv::Size(kernelSize, kernelSize), 0);
    return gaussianBlur;
}

// 2a. Laplacian
cv::Mat applyLaplacian(const cv::Mat& img, int kernelSize = 3)
{
    cv::Mat laplacian, laplacian_abs;
    cv::Laplacian(img, laplacian, CV_16S, kernelSize);
    cv::convertScaleAbs(laplacian, laplacian_abs);
    return laplacian_abs;
}

// 2a. Sobel (ambele direcții)
cv::Mat applySobel(const cv::Mat& img, int kernelSize = 3)
{
    cv::Mat sobel_x, sobel_y, sobel_abs;
    cv::Sobel(img, sobel_x, CV_16S, 1, 0, kernelSize);
    cv::Sobel(img, sobel_y, CV_16S, 0, 1, kernelSize);
    cv::convertScaleAbs(sobel_x, sobel_x);
    cv::convertScaleAbs(sobel_y, sobel_y);
    cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, sobel_abs);
    return sobel_abs;
}

// 2a. Mean Blur (media aritmetică)
cv::Mat applyMeanBlur(const cv::Mat& img, int kernelSize = 5)
{
    cv::Mat meanBlur;
    cv::blur(img, meanBlur, cv::Size(kernelSize, kernelSize));
    return meanBlur;
}

// 2b. Detectare muchii pe orizontală
cv::Mat detectHorizontalEdges(const cv::Mat& img, int kernelSize = 3)
{
    cv::Mat sobel_horizontal, sobel_horizontal_abs;
    cv::Sobel(img, sobel_horizontal, CV_16S, 1, 0, kernelSize);
    cv::convertScaleAbs(sobel_horizontal, sobel_horizontal_abs);
    return sobel_horizontal_abs;
}

// 2b. Detectare muchii pe verticală
cv::Mat detectVerticalEdges(const cv::Mat& img, int kernelSize = 3)
{
    cv::Mat sobel_vertical, sobel_vertical_abs;
    cv::Sobel(img, sobel_vertical, CV_16S, 0, 1, kernelSize);
    cv::convertScaleAbs(sobel_vertical, sobel_vertical_abs);
    return sobel_vertical_abs;
}

// 2b. Combinare muchii (orizontale + verticale)
cv::Mat combineEdges(const cv::Mat& horizontal, const cv::Mat& vertical)
{
    cv::Mat combined;
    cv::addWeighted(horizontal, 0.5, vertical, 0.5, 0, combined);
    return combined;
}

// 2c. Median Blur
cv::Mat applyMedianBlur(const cv::Mat& img, int kernelSize = 3)
{
    cv::Mat medianBlur;
    cv::medianBlur(img, medianBlur, kernelSize);
    return medianBlur;
}

// ==================== APLICAȚIA 1 ====================

void aplicatia1a() {
    std::cout << "\n=== APLICAȚIA 1a: Negativare și Binarizare ===" << std::endl;

    // Load image in grayscale
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }

    // Negativare folosind funcția helper
    unsigned char* negatedData = negateImage(img.data, img.cols, img.rows);
    cv::Mat negatedImg(img.rows, img.cols, CV_8UC1, negatedData);

    // Binarizare folosind funcția helper
    cv::Mat binaryImg = binarizeImage(img, 128);

    // Save results
    cv::imwrite("output/1a_original.png", img);
    cv::imwrite("output/1a_negativare.png", negatedImg);
    cv::imwrite("output/1a_binarizare.png", binaryImg);

    std::cout << "✓ Salvat: output/1a_original.png" << std::endl;
    std::cout << "✓ Salvat: output/1a_negativare.png" << std::endl;
    std::cout << "✓ Salvat: output/1a_binarizare.png" << std::endl;

    delete[] negatedData;
}

void aplicatia1b() {
    std::cout << "\n=== APLICAȚIA 1b: Histograma și Egalizare ===" << std::endl;

    // Load image in grayscale
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }

    // Egalizare histogramă folosind funcția helper
    cv::Mat equalizedImg = equalizeHistogram(img);

    // Calculare histograme folosind funcțiile helper
    cv::Mat hist_original = calculateHistogram(img);
    cv::Mat hist_equalized = calculateHistogram(equalizedImg);

    // Desenare histograme folosind funcția helper
    cv::Mat histImage_original = drawHistogram(hist_original);
    cv::Mat histImage_equalized = drawHistogram(hist_equalized);

    // Save results
    cv::imwrite("output/1b_original.png", img);
    cv::imwrite("output/1b_histograma_originala.png", histImage_original);
    cv::imwrite("output/1b_egalizata.png", equalizedImg);
    cv::imwrite("output/1b_histograma_egalizata.png", histImage_equalized);

    std::cout << "✓ Salvat: output/1b_original.png" << std::endl;
    std::cout << "✓ Salvat: output/1b_histograma_originala.png" << std::endl;
    std::cout << "✓ Salvat: output/1b_egalizata.png" << std::endl;
    std::cout << "✓ Salvat: output/1b_histograma_egalizata.png" << std::endl;
}

// ==================== APLICAȚIA 2 ====================

void aplicatia2a() {
    std::cout << "\n=== APLICAȚIA 2a: Filtre Spațiale ===" << std::endl;

    // Load image
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }

    // Aplicare filtre folosind funcțiile helper
    cv::Mat motionBlur = applyMotionBlur(img, 5);
    cv::Mat gaussianBlur = applyGaussianBlur(img, 5);
    cv::Mat laplacian_abs = applyLaplacian(img, 3);
    cv::Mat sobel_abs = applySobel(img, 3);
    cv::Mat meanBlur = applyMeanBlur(img, 5);

    // Save results
    cv::imwrite("output/2a_original.png", img);
    cv::imwrite("output/2a_motion_blur.png", motionBlur);
    cv::imwrite("output/2a_gaussian_blur.png", gaussianBlur);
    cv::imwrite("output/2a_laplacian.png", laplacian_abs);
    cv::imwrite("output/2a_sobel.png", sobel_abs);
    cv::imwrite("output/2a_mean_blur.png", meanBlur);

    std::cout << "✓ Salvat: output/2a_original.png" << std::endl;
    std::cout << "✓ Salvat: output/2a_motion_blur.png" << std::endl;
    std::cout << "✓ Salvat: output/2a_gaussian_blur.png" << std::endl;
    std::cout << "✓ Salvat: output/2a_laplacian.png" << std::endl;
    std::cout << "✓ Salvat: output/2a_sobel.png" << std::endl;
    std::cout << "✓ Salvat: output/2a_mean_blur.png" << std::endl;
}

void aplicatia2b() {
    std::cout << "\n=== APLICAȚIA 2b: Identificarea Muchiilor ===" << std::endl;

    // Load image
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }

    // Detectare muchii folosind funcțiile helper
    cv::Mat sobel_horizontal_abs = detectHorizontalEdges(img, 3);
    cv::Mat sobel_vertical_abs = detectVerticalEdges(img, 3);
    cv::Mat edges_combined = combineEdges(sobel_horizontal_abs, sobel_vertical_abs);

    // Save results
    cv::imwrite("output/2b_original.png", img);
    cv::imwrite("output/2b_muchii_orizontale.png", sobel_horizontal_abs);
    cv::imwrite("output/2b_muchii_verticale.png", sobel_vertical_abs);
    cv::imwrite("output/2b_muchii_combinate.png", edges_combined);

    std::cout << "✓ Salvat: output/2b_original.png" << std::endl;
    std::cout << "✓ Salvat: output/2b_muchii_orizontale.png" << std::endl;
    std::cout << "✓ Salvat: output/2b_muchii_verticale.png" << std::endl;
    std::cout << "✓ Salvat: output/2b_muchii_combinate.png" << std::endl;
}

void aplicatia2c() {
    std::cout << "\n=== APLICAȚIA 2c: Filtru Median ===" << std::endl;

    // Load image
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }

    // Aplicare filtru median folosind funcția helper
    cv::Mat median3x3 = applyMedianBlur(img, 3);
    cv::Mat median5x5 = applyMedianBlur(img, 5);
    cv::Mat median7x7 = applyMedianBlur(img, 7);

    // Save results
    cv::imwrite("output/2c_original.png", img);
    cv::imwrite("output/2c_median_3x3.png", median3x3);
    cv::imwrite("output/2c_median_5x5.png", median5x5);
    cv::imwrite("output/2c_median_7x7.png", median7x7);

    std::cout << "✓ Salvat: output/2c_original.png" << std::endl;
    std::cout << "✓ Salvat: output/2c_median_3x3.png" << std::endl;
    std::cout << "✓ Salvat: output/2c_median_5x5.png" << std::endl;
    std::cout << "✓ Salvat: output/2c_median_7x7.png" << std::endl;
}

// ==================== MAIN ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Prelucrarea Imaginilor - Laborator 7" << std::endl;
    std::cout << "Versiune salvare în fișiere" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create output directory
    system("mkdir -p output");

    int choice;

    while(true) {
        std::cout << "\nMeniu:" << std::endl;
        std::cout << "1. Aplicația 1a - Negativare și Binarizare" << std::endl;
        std::cout << "2. Aplicația 1b - Histograma și Egalizare" << std::endl;
        std::cout << "3. Aplicația 2a - Filtre Spațiale" << std::endl;
        std::cout << "4. Aplicația 2b - Identificarea Muchiilor" << std::endl;
        std::cout << "5. Aplicația 2c - Filtru Median" << std::endl;
        std::cout << "6. Rulează toate aplicațiile" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "Alege opțiunea: ";
        std::cin >> choice;

        switch(choice) {
            case 1:
                aplicatia1a();
                break;
            case 2:
                aplicatia1b();
                break;
            case 3:
                aplicatia2a();
                break;
            case 4:
                aplicatia2b();
                break;
            case 5:
                aplicatia2c();
                break;
            case 6:
                std::cout << "\n=== Rulare toate aplicațiile ===" << std::endl;
                aplicatia1a();
                aplicatia1b();
                aplicatia2a();
                aplicatia2b();
                aplicatia2c();
                std::cout << "\n✓ Toate aplicațiile au fost executate!" << std::endl;
                std::cout << "✓ Verifică directorul 'output/' pentru rezultate." << std::endl;
                break;
            case 0:
                std::cout << "Exiting..." << std::endl;
                return 0;
            default:
                std::cout << "Opțiune invalidă!" << std::endl;
        }
    }

    return 0;
}
