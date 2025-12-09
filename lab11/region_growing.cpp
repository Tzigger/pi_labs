/**
 * Laborator 11 - Region Growing Implementation
 */

#include "region_growing.h"
#include <queue>
#include <iostream>

using namespace cv;
using namespace std;

// Conversie la grayscale
Mat toGrayscale(const Mat& image) {
    if (image.channels() == 1) {
        return image.clone();
    }
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    return gray;
}

// Afișare statistici imagine
void printImageStats(const Mat& image, const string& name) {
    cout << "=== Statistici pentru " << name << " ===\n";
    cout << "Dimensiuni: " << image.cols << " x " << image.rows << endl;
    cout << "Canale: " << image.channels() << endl;
    
    if (image.channels() == 1) {
        double minVal, maxVal;
        minMaxLoc(image, &minVal, &maxVal);
        Scalar mean = cv::mean(image);
        cout << "Min: " << (int)minVal << ", Max: " << (int)maxVal << endl;
        cout << "Medie: " << mean[0] << endl;
    }
    cout << endl;
}

// =============================================================================
// EXERCIȚIUL 1: Region Growing Recursiv (4-conectat)
// =============================================================================
void recursive_region_growing_4(int x, int y, Mat& segmMask, const Mat& image, 
                               uchar seedValue, int threshold) {
    int w = image.cols;
    int h = image.rows;
    
    // Verificăm limitele imaginii și dacă pixelul a fost deja procesat
    if (x < 0 || y < 0 || x >= w || y >= h || segmMask.at<uchar>(y, x) == 255) {
        return;
    }
    
    // Verificăm criteriul de similaritate
    if (abs(image.at<uchar>(y, x) - seedValue) > threshold) {
        return;
    }
    
    // Marcăm pixelul ca făcând parte din regiune
    segmMask.at<uchar>(y, x) = 255;
    
    // Recursie pentru cei 4 vecini
    recursive_region_growing_4(x - 1, y, segmMask, image, seedValue, threshold);  // Stânga
    recursive_region_growing_4(x + 1, y, segmMask, image, seedValue, threshold);  // Dreapta
    recursive_region_growing_4(x, y - 1, segmMask, image, seedValue, threshold);  // Sus
    recursive_region_growing_4(x, y + 1, segmMask, image, seedValue, threshold);  // Jos
}

// =============================================================================
// EXERCIȚIUL 2: Region Growing Recursiv (8-conectat)
// =============================================================================
void recursive_region_growing_8(int x, int y, Mat& segmMask, const Mat& image,
                               uchar seedValue, int threshold) {
    int w = image.cols;
    int h = image.rows;
    
    // Verificăm limitele imaginii și dacă pixelul a fost deja procesat
    if (x < 0 || y < 0 || x >= w || y >= h || segmMask.at<uchar>(y, x) == 255) {
        return;
    }
    
    // Verificăm criteriul de similaritate
    if (abs(image.at<uchar>(y, x) - seedValue) > threshold) {
        return;
    }
    
    // Marcăm pixelul ca făcând parte din regiune
    segmMask.at<uchar>(y, x) = 255;
    
    // Recursie pentru cei 8 vecini
    // Vecini horizontali și verticali
    recursive_region_growing_8(x - 1, y, segmMask, image, seedValue, threshold);      // Stânga
    recursive_region_growing_8(x + 1, y, segmMask, image, seedValue, threshold);      // Dreapta
    recursive_region_growing_8(x, y - 1, segmMask, image, seedValue, threshold);      // Sus
    recursive_region_growing_8(x, y + 1, segmMask, image, seedValue, threshold);      // Jos
    
    // Vecini diagonali
    recursive_region_growing_8(x - 1, y - 1, segmMask, image, seedValue, threshold);  // Stânga-sus
    recursive_region_growing_8(x + 1, y - 1, segmMask, image, seedValue, threshold);  // Dreapta-sus
    recursive_region_growing_8(x - 1, y + 1, segmMask, image, seedValue, threshold);  // Stânga-jos
    recursive_region_growing_8(x + 1, y + 1, segmMask, image, seedValue, threshold);  // Dreapta-jos
}

// =============================================================================
// EXERCIȚIUL 3: Region Growing folosind cv::floodFill
// =============================================================================
Mat floodFill_region_growing(const Mat& image, int x, int y, int threshold) {
    Mat result = image.clone();
    Mat mask = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
    
    // Obținem valoarea seed-ului
    uchar seedValue = image.at<uchar>(y, x);
    
    // Flags: 4-conectat | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY
    // FLOODFILL_FIXED_RANGE: compară cu seedValue original, nu cu vecinii
    int flags = 4 | FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | (255 << 8);
    
    // Aplicăm floodFill
    Rect rect;
    cv::floodFill(result, mask, Point(x, y), Scalar(255), &rect, 
                  Scalar(threshold), Scalar(threshold), flags);
    
    // Extragem masca (excludem border-ul de 1 pixel)
    Mat segmMask = mask(Rect(1, 1, image.cols, image.rows));
    
    return segmMask;
}

// =============================================================================
// EXERCIȚIUL 4: Multi-seed Region Growing
// =============================================================================
Mat multi_seed_region_growing(const Mat& image, const vector<SeedPoint>& seeds, bool useFloodFill) {
    Mat segmMask = Mat::zeros(image.size(), CV_8UC1);
    
    for (size_t i = 0; i < seeds.size(); i++) {
        const SeedPoint& seed = seeds[i];
        
        cout << "Procesare seed " << (i + 1) << ": "
             << "x=" << seed.x << ", y=" << seed.y 
             << ", seedValue=" << (int)seed.seedValue
             << ", threshold=" << seed.threshold << endl;
        
        if (useFloodFill) {
            // Folosim floodFill
            Mat tempMask = floodFill_region_growing(image, seed.x, seed.y, seed.threshold);
            
            // Combinăm cu masca existentă (OR logic)
            bitwise_or(segmMask, tempMask, segmMask);
        } else {
            // Folosim funcția recursivă (8-conectat)
            recursive_region_growing_8(seed.x, seed.y, segmMask, image, 
                                      seed.seedValue, seed.threshold);
        }
    }
    
    return segmMask;
}

// =============================================================================
// Funcții de vizualizare
// =============================================================================
Mat visualizeRegions(const Mat& original, const Mat& mask, const vector<SeedPoint>& seeds) {
    Mat display;
    
    // Convertim originalul la color
    if (original.channels() == 1) {
        cvtColor(original, display, COLOR_GRAY2BGR);
    } else {
        display = original.clone();
    }
    
    // Suprapunem masca cu o culoare semi-transparentă
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) == 255) {
                // Overlay verde semi-transparent
                display.at<Vec3b>(y, x)[0] = display.at<Vec3b>(y, x)[0] * 0.5;
                display.at<Vec3b>(y, x)[1] = display.at<Vec3b>(y, x)[1] * 0.5 + 127;
                display.at<Vec3b>(y, x)[2] = display.at<Vec3b>(y, x)[2] * 0.5;
            }
        }
    }
    
    // Desenăm punctele seed
    for (const auto& seed : seeds) {
        circle(display, Point(seed.x, seed.y), 5, seed.color, -1);
        circle(display, Point(seed.x, seed.y), 6, Scalar(0, 0, 0), 1);
        
        // Adăugăm label
        string label = "S" + to_string(&seed - &seeds[0] + 1);
        putText(display, label, Point(seed.x + 10, seed.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
        putText(display, label, Point(seed.x + 10, seed.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, seed.color, 1);
    }
    
    return display;
}

Mat createComparison(const Mat& original, const Mat& recursive4, 
                    const Mat& recursive8, const Mat& floodFill) {
    // Convertim toate la BGR pentru afișare
    Mat origColor, rec4Color, rec8Color, ffColor;
    
    if (original.channels() == 1) {
        cvtColor(original, origColor, COLOR_GRAY2BGR);
    } else {
        origColor = original.clone();
    }
    
    cvtColor(recursive4, rec4Color, COLOR_GRAY2BGR);
    cvtColor(recursive8, rec8Color, COLOR_GRAY2BGR);
    cvtColor(floodFill, ffColor, COLOR_GRAY2BGR);
    
    // Adăugăm text
    putText(origColor, "Original", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(rec4Color, "Recursiv 4-conectat", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(rec8Color, "Recursiv 8-conectat", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(ffColor, "FloodFill", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    
    // Creăm grid 2x2
    Mat row1, row2, result;
    hconcat(origColor, rec4Color, row1);
    hconcat(rec8Color, ffColor, row2);
    vconcat(row1, row2, result);
    
    return result;
}
