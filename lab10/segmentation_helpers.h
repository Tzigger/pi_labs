/**
 * Laborator 10 - Segmentarea Imaginilor (1)
 * Header pentru funcțiile de segmentare
 */

#ifndef SEGMENTATION_HELPERS_H
#define SEGMENTATION_HELPERS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

// =============================================================================
// 1. SEGMENTARE CU PRAG GLOBAL (Global Thresholding)
// =============================================================================

/**
 * Segmentare cu prag simplu (binarizare)
 * g(x,y) = 1 dacă f(x,y) >= T, altfel 0
 * 
 * @param image Imaginea de intrare (grayscale)
 * @param threshold Valoarea pragului T
 * @param objectValue Valoarea pentru obiecte (default 255)
 * @param backgroundValue Valoarea pentru fundal (default 0)
 * @return Imaginea binarizată
 */
Mat applyGlobalThreshold(const Mat& image, int threshold, 
                         int objectValue = 255, int backgroundValue = 0);

/**
 * Segmentare cu prag de bandă
 * g(x,y) = 1 dacă T1 <= f(x,y) <= T2, altfel 0
 * 
 * @param image Imaginea de intrare
 * @param lowThreshold Pragul inferior T1
 * @param highThreshold Pragul superior T2
 * @return Imaginea binarizată
 */
Mat applyBandThreshold(const Mat& image, int lowThreshold, int highThreshold);

/**
 * Segmentare cu praguri multiple
 * Returnează o imagine cu mai multe nivele (regiuni diferite)
 * 
 * @param image Imaginea de intrare
 * @param thresholds Vector de praguri [T1, T2, ..., Tn]
 * @return Imaginea segmentată cu niveluri multiple
 */
Mat applyMultipleThresholds(const Mat& image, const vector<int>& thresholds);

/**
 * Semiprag (Semi-threshold) - mascarea fundalului
 * g(x,y) = f(x,y) dacă f(x,y) >= T, altfel 0
 * 
 * @param image Imaginea de intrare
 * @param threshold Pragul T
 * @return Imaginea cu fundal mascat
 */
Mat applySemiThreshold(const Mat& image, int threshold);

// =============================================================================
// 2. METODE DE SELECTARE A PRAGULUI
// =============================================================================

/**
 * Calculează histograma unei imagini
 * 
 * @param image Imaginea de intrare (grayscale)
 * @return Histograma (256 nivele)
 */
Mat computeHistogram(const Mat& image);

/**
 * Afișează histograma ca imagine
 * 
 * @param histogram Histograma calculată
 * @param histImage Imaginea de ieșire pentru histogramă
 * @param width Lățimea imaginii histogramei
 * @param height Înălțimea imaginii histogramei
 */
void drawHistogram(const Mat& histogram, Mat& histImage, int width = 512, int height = 400);

/**
 * Metoda Otsu pentru selectarea automată a pragului
 * Minimizează varianța intra-clasă
 * 
 * @param image Imaginea de intrare
 * @return Pragul optimal calculat
 */
int computeOtsuThreshold(const Mat& image);

/**
 * Metoda iterativă pentru selectarea pragului
 * 
 * @param image Imaginea de intrare
 * @param epsilon Criteriul de convergență
 * @return Pragul calculat iterativ
 */
int computeIterativeThreshold(const Mat& image, double epsilon = 0.5);

/**
 * Găsește minimele locale în histogramă (pentru praguri multiple)
 * 
 * @param histogram Histograma
 * @param smoothing Dimensiunea ferestrei de netezire
 * @return Vector cu pozițiile minimelor locale
 */
vector<int> findHistogramMinima(const Mat& histogram, int smoothing = 5);

// =============================================================================
// 3. SEGMENTARE LOCALĂ (Adaptive Thresholding)
// =============================================================================

/**
 * Segmentare cu prag adaptiv (local)
 * Pragul variază în funcție de vecinătatea locală
 * 
 * @param image Imaginea de intrare
 * @param blockSize Dimensiunea blocului pentru calculul pragului local
 * @param C Constantă scăzută din media locală
 * @param method 0 = media, 1 = gaussian
 * @return Imaginea binarizată adaptiv
 */
Mat applyAdaptiveThreshold(const Mat& image, int blockSize, double C, int method = 0);

// =============================================================================
// 4. UTILITĂȚI
// =============================================================================

/**
 * Convertește o imagine color la grayscale
 */
Mat toGrayscale(const Mat& image);

/**
 * Afișează statistici despre imagine
 */
void printImageStats(const Mat& image, const string& name);

/**
 * Creează o imagine cu mai multe rezultate pentru comparație
 */
Mat createComparisonImage(const vector<Mat>& images, const vector<string>& titles, int cols = 3);

#endif // SEGMENTATION_HELPERS_H
