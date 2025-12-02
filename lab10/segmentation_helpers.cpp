/**
 * Laborator 10 - Segmentarea Imaginilor (1)
 * Implementarea funcțiilor de segmentare
 */

#include "segmentation_helpers.h"
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// =============================================================================
// 1. SEGMENTARE CU PRAG GLOBAL
// =============================================================================

Mat applyGlobalThreshold(const Mat& image, int threshold, 
                         int objectValue, int backgroundValue) {
    Mat gray = toGrayscale(image);
    Mat result(gray.size(), CV_8UC1);
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            // g(x,y) = objectValue dacă f(x,y) >= T, altfel backgroundValue
            result.at<uchar>(y, x) = (pixel >= threshold) ? objectValue : backgroundValue;
        }
    }
    
    return result;
}

Mat applyBandThreshold(const Mat& image, int lowThreshold, int highThreshold) {
    Mat gray = toGrayscale(image);
    Mat result(gray.size(), CV_8UC1);
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            // g(x,y) = 255 dacă T1 <= f(x,y) <= T2, altfel 0
            result.at<uchar>(y, x) = (pixel >= lowThreshold && pixel <= highThreshold) ? 255 : 0;
        }
    }
    
    return result;
}

Mat applyMultipleThresholds(const Mat& image, const vector<int>& thresholds) {
    Mat gray = toGrayscale(image);
    Mat result(gray.size(), CV_8UC1);
    
    // Calculăm nivelurile de gri pentru fiecare regiune
    int numRegions = thresholds.size() + 1;
    vector<int> regionValues(numRegions);
    
    for (int i = 0; i < numRegions; i++) {
        regionValues[i] = (255 * i) / (numRegions - 1);
    }
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            
            // Găsim regiunea corespunzătoare
            int region = 0;
            for (size_t i = 0; i < thresholds.size(); i++) {
                if (pixel >= thresholds[i]) {
                    region = i + 1;
                }
            }
            
            result.at<uchar>(y, x) = regionValues[region];
        }
    }
    
    return result;
}

Mat applySemiThreshold(const Mat& image, int threshold) {
    Mat gray = toGrayscale(image);
    Mat result(gray.size(), CV_8UC1);
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            // g(x,y) = f(x,y) dacă f(x,y) >= T, altfel 0
            result.at<uchar>(y, x) = (pixel >= threshold) ? pixel : 0;
        }
    }
    
    return result;
}

// =============================================================================
// 2. METODE DE SELECTARE A PRAGULUI
// =============================================================================

Mat computeHistogram(const Mat& image) {
    Mat gray = toGrayscale(image);
    
    // Calculăm histograma manual
    Mat histogram = Mat::zeros(1, 256, CV_32F);
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            histogram.at<float>(0, pixel)++;
        }
    }
    
    return histogram;
}

void drawHistogram(const Mat& histogram, Mat& histImage, int width, int height) {
    histImage = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
    
    // Găsim valoarea maximă pentru normalizare
    double maxVal;
    minMaxLoc(histogram, nullptr, &maxVal);
    
    int binWidth = cvRound((double)width / 256);
    
    // Desenăm barele histogramei
    for (int i = 0; i < 256; i++) {
        float binVal = histogram.at<float>(0, i);
        int barHeight = cvRound((binVal / maxVal) * (height - 20));
        
        rectangle(histImage,
                  Point(i * binWidth, height - barHeight),
                  Point((i + 1) * binWidth - 1, height),
                  Scalar(100, 100, 100),
                  FILLED);
    }
    
    // Adăugăm axa X
    line(histImage, Point(0, height - 1), Point(width - 1, height - 1), Scalar(0, 0, 0), 1);
}

int computeOtsuThreshold(const Mat& image) {
    Mat gray = toGrayscale(image);
    Mat histogram = computeHistogram(gray);
    
    int totalPixels = gray.rows * gray.cols;
    
    // Calculăm suma totală
    double sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += i * histogram.at<float>(0, i);
    }
    
    double sumB = 0;
    int wB = 0;
    int wF = 0;
    
    double maxVariance = 0;
    int optimalThreshold = 0;
    
    for (int t = 0; t < 256; t++) {
        wB += (int)histogram.at<float>(0, t);  // Weight Background
        if (wB == 0) continue;
        
        wF = totalPixels - wB;  // Weight Foreground
        if (wF == 0) break;
        
        sumB += t * histogram.at<float>(0, t);
        
        double mB = sumB / wB;           // Mean Background
        double mF = (sum - sumB) / wF;   // Mean Foreground
        
        // Varianța între clase
        double variance = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }
    
    return optimalThreshold;
}

int computeIterativeThreshold(const Mat& image, double epsilon) {
    Mat gray = toGrayscale(image);
    
    // Inițializăm pragul cu media imaginii
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    double T = (minVal + maxVal) / 2.0;
    
    double Tprev;
    int maxIterations = 100;
    int iteration = 0;
    
    do {
        Tprev = T;
        
        double sumBelow = 0, countBelow = 0;
        double sumAbove = 0, countAbove = 0;
        
        for (int y = 0; y < gray.rows; y++) {
            for (int x = 0; x < gray.cols; x++) {
                uchar pixel = gray.at<uchar>(y, x);
                if (pixel < T) {
                    sumBelow += pixel;
                    countBelow++;
                } else {
                    sumAbove += pixel;
                    countAbove++;
                }
            }
        }
        
        double m1 = (countBelow > 0) ? sumBelow / countBelow : 0;
        double m2 = (countAbove > 0) ? sumAbove / countAbove : 0;
        
        T = (m1 + m2) / 2.0;
        iteration++;
        
    } while (abs(T - Tprev) > epsilon && iteration < maxIterations);
    
    cout << "Prag iterativ calculat în " << iteration << " iterații: " << (int)T << endl;
    
    return (int)T;
}

vector<int> findHistogramMinima(const Mat& histogram, int smoothing) {
    vector<int> minima;
    
    // Netezim histograma
    Mat smoothed;
    if (smoothing > 0) {
        GaussianBlur(histogram, smoothed, Size(smoothing * 2 + 1, 1), smoothing / 2.0);
    } else {
        smoothed = histogram.clone();
    }
    
    // Găsim minimele locale
    for (int i = 10; i < 245; i++) {  // Evităm extremitățile
        float prev = smoothed.at<float>(0, i - 1);
        float curr = smoothed.at<float>(0, i);
        float next = smoothed.at<float>(0, i + 1);
        
        if (curr < prev && curr < next && curr > 0) {
            minima.push_back(i);
        }
    }
    
    return minima;
}

// =============================================================================
// 3. SEGMENTARE LOCALĂ (Adaptive Thresholding)
// =============================================================================

Mat applyAdaptiveThreshold(const Mat& image, int blockSize, double C, int method) {
    Mat gray = toGrayscale(image);
    Mat result(gray.size(), CV_8UC1);
    
    int halfBlock = blockSize / 2;
    
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            // Calculăm media locală
            double sum = 0;
            int count = 0;
            
            for (int dy = -halfBlock; dy <= halfBlock; dy++) {
                for (int dx = -halfBlock; dx <= halfBlock; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    
                    if (ny >= 0 && ny < gray.rows && nx >= 0 && nx < gray.cols) {
                        if (method == 0) {
                            // Media aritmetică
                            sum += gray.at<uchar>(ny, nx);
                            count++;
                        } else {
                            // Ponderare gaussiană simplificată
                            double weight = exp(-(dx*dx + dy*dy) / (2.0 * halfBlock * halfBlock));
                            sum += gray.at<uchar>(ny, nx) * weight;
                            count++;
                        }
                    }
                }
            }
            
            double localThreshold = (sum / count) - C;
            uchar pixel = gray.at<uchar>(y, x);
            result.at<uchar>(y, x) = (pixel > localThreshold) ? 255 : 0;
        }
    }
    
    return result;
}

// =============================================================================
// 4. UTILITĂȚI
// =============================================================================

Mat toGrayscale(const Mat& image) {
    if (image.channels() == 1) {
        return image.clone();
    }
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    return gray;
}

void printImageStats(const Mat& image, const string& name) {
    Mat gray = toGrayscale(image);
    
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    
    Scalar mean = cv::mean(gray);
    
    cout << "=== Statistici pentru " << name << " ===" << endl;
    cout << "Dimensiuni: " << image.cols << " x " << image.rows << endl;
    cout << "Canale: " << image.channels() << endl;
    cout << "Min: " << minVal << ", Max: " << maxVal << endl;
    cout << "Medie: " << mean[0] << endl;
}

Mat createComparisonImage(const vector<Mat>& images, const vector<string>& titles, int cols) {
    if (images.empty()) return Mat();
    
    int rows = (images.size() + cols - 1) / cols;
    int cellWidth = 300;
    int cellHeight = 250;
    int titleHeight = 30;
    
    Mat result(rows * (cellHeight + titleHeight), cols * cellWidth, CV_8UC3, Scalar(255, 255, 255));
    
    for (size_t i = 0; i < images.size(); i++) {
        int row = i / cols;
        int col = i % cols;
        
        int x = col * cellWidth;
        int y = row * (cellHeight + titleHeight);
        
        // Redimensionăm imaginea
        Mat resized;
        Mat img = images[i];
        if (img.channels() == 1) {
            cvtColor(img, img, COLOR_GRAY2BGR);
        }
        resize(img, resized, Size(cellWidth - 10, cellHeight - 10));
        
        // Copiem imaginea
        resized.copyTo(result(Rect(x + 5, y + titleHeight, cellWidth - 10, cellHeight - 10)));
        
        // Adăugăm titlul
        if (i < titles.size()) {
            putText(result, titles[i], Point(x + 10, y + 20), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }
    }
    
    return result;
}
