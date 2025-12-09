/**
 * Laborator 11 - Region Growing Helper Functions
 */

#ifndef REGION_GROWING_H
#define REGION_GROWING_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Structură pentru seed point
struct SeedPoint {
    int x;
    int y;
    uchar seedValue;
    int threshold;
    Scalar color;  // Culoarea pentru vizualizare
    
    SeedPoint(int _x, int _y, uchar _val, int _thresh, Scalar _color = Scalar(0, 255, 0))
        : x(_x), y(_y), seedValue(_val), threshold(_thresh), color(_color) {}
};

// Funcții pentru conversie la grayscale și afișare
Mat toGrayscale(const Mat& image);
void printImageStats(const Mat& image, const string& name);

// Exercițiul 1: Region growing recursiv (4-conectat)
void recursive_region_growing_4(int x, int y, Mat& segmMask, const Mat& image, 
                               uchar seedValue, int threshold);

// Exercițiul 2: Region growing recursiv (8-conectat)
void recursive_region_growing_8(int x, int y, Mat& segmMask, const Mat& image,
                               uchar seedValue, int threshold);

// Exercițiul 3: Region growing folosind cv::floodFill
Mat floodFill_region_growing(const Mat& image, int x, int y, int threshold);

// Exercițiul 4: Multi-seed region growing
Mat multi_seed_region_growing(const Mat& image, const vector<SeedPoint>& seeds, bool useFloodFill = true);

// Funcții helper pentru vizualizare
Mat visualizeRegions(const Mat& original, const Mat& mask, const vector<SeedPoint>& seeds);
Mat createComparison(const Mat& original, const Mat& recursive4, const Mat& recursive8, const Mat& floodFill);

#endif // REGION_GROWING_H
