/**
 * Laborator 11 - Segmentarea Imaginilor (2)
 * Region Growing
 * 
 * Exerciții:
 * 1. Region growing recursiv (4-conectat)
 * 2. Region growing recursiv (8-conectat)
 * 3. Region growing folosind cv::floodFill
 * 4. Multi-seed region growing
 * 5. Aplicație interactivă cu mouse
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include "region_growing.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Variabile globale pentru mouse callback
Mat g_displayImage;
Mat g_originalImage;
Mat g_grayImage;
vector<SeedPoint> g_seeds;
int g_currentThreshold = 40;

void displayMenu() {
    cout << "\n====== LABORATOR 11 - Region Growing ======\n";
    cout << "1. Region Growing Recursiv (4-conectat)\n";
    cout << "2. Region Growing Recursiv (8-conectat)\n";
    cout << "3. Region Growing cu cv::floodFill\n";
    cout << "4. Multi-seed Region Growing (3 seeds predefinite)\n";
    cout << "5. Comparație toate metodele (1 seed)\n";
    cout << "6. Aplicație interactivă (click pentru seed)\n";
    cout << "7. Rulează TOATE exercițiile și salvează\n";
    cout << "0. Ieșire\n";
    cout << "============================================\n";
    cout << "Alegeți opțiunea: ";
}

// =============================================================================
// Mouse callback pentru aplicația interactivă
// =============================================================================
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        cout << "\nClick la poziția: x=" << x << ", y=" << y << endl;
        
        if (g_grayImage.empty()) {
            cout << "Eroare: Imaginea nu a fost încărcată!\n";
            return;
        }
        
        uchar seedValue = g_grayImage.at<uchar>(y, x);
        cout << "Valoare seed: " << (int)seedValue << endl;
        cout << "Threshold curent: " << g_currentThreshold << endl;
        
        // Aplicăm region growing
        Mat mask = floodFill_region_growing(g_grayImage, x, y, g_currentThreshold);
        
        // Calculăm numărul de pixeli din regiune
        int pixelCount = countNonZero(mask);
        cout << "Pixeli în regiune: " << pixelCount << endl;
        
        // Actualizăm afișarea
        g_displayImage = g_originalImage.clone();
        if (g_displayImage.channels() == 1) {
            cvtColor(g_displayImage, g_displayImage, COLOR_GRAY2BGR);
        }
        
        // Colorăm regiunea
        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                if (mask.at<uchar>(i, j) == 255) {
                    g_displayImage.at<Vec3b>(i, j)[0] = g_displayImage.at<Vec3b>(i, j)[0] * 0.5;
                    g_displayImage.at<Vec3b>(i, j)[1] = g_displayImage.at<Vec3b>(i, j)[1] * 0.5 + 127;
                    g_displayImage.at<Vec3b>(i, j)[2] = g_displayImage.at<Vec3b>(i, j)[2] * 0.5;
                }
            }
        }
        
        // Marcăm punctul seed
        circle(g_displayImage, Point(x, y), 5, Scalar(0, 0, 255), -1);
        circle(g_displayImage, Point(x, y), 6, Scalar(255, 255, 255), 2);
        
        // Afișăm informații
        string info = "Seed: (" + to_string(x) + "," + to_string(y) + 
                     ") Val=" + to_string((int)seedValue) + 
                     " T=" + to_string(g_currentThreshold) +
                     " Pixeli=" + to_string(pixelCount);
        putText(g_displayImage, info, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 3);
        putText(g_displayImage, info, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
        
        imshow("Interactive Region Growing", g_displayImage);
    }
}

// =============================================================================
// EXERCIȚIUL 1: Region Growing Recursiv (4-conectat)
// =============================================================================
void exercitiul1(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 1: Region Growing Recursiv (4-conectat) ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Seed point de test: centrul imaginii
    int x = gray.cols / 2;
    int y = gray.rows / 2;
    uchar seedValue = gray.at<uchar>(y, x);
    int threshold = 40;
    
    cout << "Seed: x=" << x << ", y=" << y << endl;
    cout << "SeedValue: " << (int)seedValue << ", Threshold: " << threshold << endl;
    
    // Aplicăm region growing
    Mat mask = Mat::zeros(gray.size(), CV_8UC1);
    recursive_region_growing_4(x, y, mask, gray, seedValue, threshold);
    
    int pixelCount = countNonZero(mask);
    cout << "Pixeli în regiune: " << pixelCount << endl;
    
    // Vizualizare
    vector<SeedPoint> seeds = {SeedPoint(x, y, seedValue, threshold, Scalar(0, 0, 255))};
    Mat display = visualizeRegions(gray, mask, seeds);
    
    imwrite("output_lab11/ex1_recursive_4connected.png", mask);
    imwrite("output_lab11/ex1_visualization.png", display);
    
    cout << "✓ Salvat: output_lab11/ex1_recursive_4connected.png\n";
    cout << "✓ Salvat: output_lab11/ex1_visualization.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 1 - Recursiv 4-conectat", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 2: Region Growing Recursiv (8-conectat)
// =============================================================================
void exercitiul2(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 2: Region Growing Recursiv (8-conectat) ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Seed point de test
    int x = gray.cols / 2;
    int y = gray.rows / 2;
    uchar seedValue = gray.at<uchar>(y, x);
    int threshold = 40;
    
    cout << "Seed: x=" << x << ", y=" << y << endl;
    cout << "SeedValue: " << (int)seedValue << ", Threshold: " << threshold << endl;
    
    // Aplicăm region growing
    Mat mask = Mat::zeros(gray.size(), CV_8UC1);
    recursive_region_growing_8(x, y, mask, gray, seedValue, threshold);
    
    int pixelCount = countNonZero(mask);
    cout << "Pixeli în regiune: " << pixelCount << endl;
    
    // Vizualizare
    vector<SeedPoint> seeds = {SeedPoint(x, y, seedValue, threshold, Scalar(255, 0, 0))};
    Mat display = visualizeRegions(gray, mask, seeds);
    
    imwrite("output_lab11/ex2_recursive_8connected.png", mask);
    imwrite("output_lab11/ex2_visualization.png", display);
    
    cout << "✓ Salvat: output_lab11/ex2_recursive_8connected.png\n";
    cout << "✓ Salvat: output_lab11/ex2_visualization.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 2 - Recursiv 8-conectat", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 3: Region Growing cu cv::floodFill
// =============================================================================
void exercitiul3(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 3: Region Growing cu cv::floodFill ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Seed point de test
    int x = gray.cols / 2;
    int y = gray.rows / 2;
    uchar seedValue = gray.at<uchar>(y, x);
    int threshold = 40;
    
    cout << "Seed: x=" << x << ", y=" << y << endl;
    cout << "SeedValue: " << (int)seedValue << ", Threshold: " << threshold << endl;
    
    // Aplicăm floodFill
    Mat mask = floodFill_region_growing(gray, x, y, threshold);
    
    int pixelCount = countNonZero(mask);
    cout << "Pixeli în regiune: " << pixelCount << endl;
    
    // Vizualizare
    vector<SeedPoint> seeds = {SeedPoint(x, y, seedValue, threshold, Scalar(0, 255, 255))};
    Mat display = visualizeRegions(gray, mask, seeds);
    
    imwrite("output_lab11/ex3_floodfill.png", mask);
    imwrite("output_lab11/ex3_visualization.png", display);
    
    cout << "✓ Salvat: output_lab11/ex3_floodfill.png\n";
    cout << "✓ Salvat: output_lab11/ex3_visualization.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 3 - FloodFill", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 4: Multi-seed Region Growing
// =============================================================================
void exercitiul4(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 4: Multi-seed Region Growing ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Trei puncte seed adaptate la dimensiunea imaginii
    // Pentru imaginea X-ray (PDF): (120, 218), (250, 215), (380, 210)
    // Pentru coins.png adaptăm coordonatele
    int w = gray.cols;
    int h = gray.rows;
    
    vector<SeedPoint> seeds;
    
    if (w > 400 && h > 300) {
        // Imagini mari (X-ray style) - folosim coordonatele din PDF
        seeds = {
            SeedPoint(120, 218, 255, 40, Scalar(0, 0, 255)),    // P1: roșu
            SeedPoint(250, 215, 255, 80, Scalar(0, 255, 0)),    // P2: verde
            SeedPoint(380, 210, 255, 110, Scalar(255, 0, 0))    // P3: albastru
        };
    } else {
        // Imagini mici (coins.png) - adaptăm la centru și colțuri
        uchar val1 = gray.at<uchar>(h/4, w/4);
        uchar val2 = gray.at<uchar>(h/2, w/2);
        uchar val3 = gray.at<uchar>(3*h/4, 3*w/4);
        
        seeds = {
            SeedPoint(w/4, h/4, val1, 40, Scalar(0, 0, 255)),       // P1: stânga-sus
            SeedPoint(w/2, h/2, val2, 60, Scalar(0, 255, 0)),       // P2: centru
            SeedPoint(3*w/4, 3*h/4, val3, 80, Scalar(255, 0, 0))    // P3: dreapta-jos
        };
    }
    
    cout << "Procesare " << seeds.size() << " puncte seed...\n";
    
    // Aplicăm multi-seed region growing cu floodFill
    Mat maskFloodFill = multi_seed_region_growing(gray, seeds, true);
    
    // Aplicăm multi-seed region growing recursiv
    Mat maskRecursive = multi_seed_region_growing(gray, seeds, false);
    
    int pixelsFloodFill = countNonZero(maskFloodFill);
    int pixelsRecursive = countNonZero(maskRecursive);
    
    cout << "\nRezultate:\n";
    cout << "FloodFill: " << pixelsFloodFill << " pixeli\n";
    cout << "Recursiv: " << pixelsRecursive << " pixeli\n";
    
    // Vizualizare
    Mat displayFF = visualizeRegions(gray, maskFloodFill, seeds);
    Mat displayRec = visualizeRegions(gray, maskRecursive, seeds);
    
    Mat comparison;
    hconcat(displayFF, displayRec, comparison);
    
    putText(comparison, "FloodFill Multi-Seed", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 3);
    putText(comparison, "FloodFill Multi-Seed", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 1);
    
    putText(comparison, "Recursiv Multi-Seed", Point(gray.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 3);
    putText(comparison, "Recursiv Multi-Seed", Point(gray.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 1);
    
    imwrite("output_lab11/ex4_multiseed_floodfill.png", maskFloodFill);
    imwrite("output_lab11/ex4_multiseed_recursive.png", maskRecursive);
    imwrite("output_lab11/ex4_comparison.png", comparison);
    
    cout << "\n✓ Salvat: output_lab11/ex4_multiseed_floodfill.png\n";
    cout << "✓ Salvat: output_lab11/ex4_multiseed_recursive.png\n";
    cout << "✓ Salvat: output_lab11/ex4_comparison.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 4 - Multi-seed", comparison);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 5: Comparație toate metodele
// =============================================================================
void exercitiul5(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 5: Comparație toate metodele ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Un singur seed pentru comparație
    int x = gray.cols / 2;
    int y = gray.rows / 2;
    uchar seedValue = gray.at<uchar>(y, x);
    int threshold = 40;
    
    cout << "Seed: x=" << x << ", y=" << y << endl;
    cout << "SeedValue: " << (int)seedValue << ", Threshold: " << threshold << endl;
    
    // Aplicăm toate metodele
    Mat mask4 = Mat::zeros(gray.size(), CV_8UC1);
    recursive_region_growing_4(x, y, mask4, gray, seedValue, threshold);
    
    Mat mask8 = Mat::zeros(gray.size(), CV_8UC1);
    recursive_region_growing_8(x, y, mask8, gray, seedValue, threshold);
    
    Mat maskFF = floodFill_region_growing(gray, x, y, threshold);
    
    cout << "\nPixeli în regiune:\n";
    cout << "  4-conectat: " << countNonZero(mask4) << endl;
    cout << "  8-conectat: " << countNonZero(mask8) << endl;
    cout << "  FloodFill:  " << countNonZero(maskFF) << endl;
    
    // Creăm comparația
    Mat comparison = createComparison(gray, mask4, mask8, maskFF);
    
    imwrite("output_lab11/ex5_comparison_all_methods.png", comparison);
    cout << "\n✓ Salvat: output_lab11/ex5_comparison_all_methods.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 5 - Comparatie Metode", comparison);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 6: Aplicație interactivă
// =============================================================================
void exercitiul6(const Mat& image, bool saveOnly = false) {
    if (saveOnly) return;  // Nu are sens în modul save-only
    
    cout << "\n=== EXERCIȚIUL 6: Aplicație interactivă ===\n";
    cout << "Click pe imagine pentru a selecta puncte seed.\n";
    cout << "Taste:\n";
    cout << "  +/- : crește/scade threshold-ul\n";
    cout << "  r   : resetează imaginea\n";
    cout << "  s   : salvează rezultatul\n";
    cout << "  ESC : ieșire\n\n";
    
    g_originalImage = image.clone();
    g_grayImage = toGrayscale(image);
    g_displayImage = g_originalImage.clone();
    g_currentThreshold = 40;
    
    if (g_displayImage.channels() == 1) {
        cvtColor(g_displayImage, g_displayImage, COLOR_GRAY2BGR);
    }
    
    namedWindow("Interactive Region Growing");
    setMouseCallback("Interactive Region Growing", mouseCallback, nullptr);
    
    cout << "Threshold curent: " << g_currentThreshold << endl;
    imshow("Interactive Region Growing", g_displayImage);
    
    while (true) {
        int key = waitKey(0);
        
        if (key == 27) {  // ESC
            break;
        } else if (key == '+' || key == '=') {
            g_currentThreshold += 5;
            cout << "Threshold: " << g_currentThreshold << endl;
        } else if (key == '-' || key == '_') {
            g_currentThreshold = max(5, g_currentThreshold - 5);
            cout << "Threshold: " << g_currentThreshold << endl;
        } else if (key == 'r' || key == 'R') {
            g_displayImage = g_originalImage.clone();
            if (g_displayImage.channels() == 1) {
                cvtColor(g_displayImage, g_displayImage, COLOR_GRAY2BGR);
            }
            cout << "Imagine resetată\n";
            imshow("Interactive Region Growing", g_displayImage);
        } else if (key == 's' || key == 'S') {
            string filename = "output_lab11/ex6_interactive_result.png";
            imwrite(filename, g_displayImage);
            cout << "✓ Salvat: " << filename << endl;
        }
    }
    
    destroyWindow("Interactive Region Growing");
}

// =============================================================================
// EXERCIȚIUL 7: Rulează toate exercițiile
// =============================================================================
void runAllExercises(const Mat& image) {
    cout << "\n========================================\n";
    cout << "  RULARE TOATE EXERCIȚIILE (1-5)\n";
    cout << "========================================\n";
    
    exercitiul1(image, true);
    exercitiul2(image, true);
    exercitiul3(image, true);
    exercitiul4(image, true);
    exercitiul5(image, true);
    
    cout << "\n========================================\n";
    cout << "  TOATE REZULTATELE AU FOST SALVATE!\n";
    cout << "  Verificați directorul: output_lab11/\n";
    cout << "========================================\n";
}

int main(int argc, char** argv) {
    cout << "============================================\n";
    cout << "   LABORATOR 11 - Region Growing           \n";
    cout << "============================================\n";
    
    // Creăm directorul de output
    fs::create_directories("output_lab11");
    
    // Încărcăm imaginea de test
    string imagePath = "Imagini_Laborator/xray.png";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    Mat image = imread(imagePath);
    if (image.empty()) {
        // Încercăm alte căi
        vector<string> altPaths = {
            "Imagini_Laborator/xray.png",
            "../lab10/Imagini_Laborator/coins.png",
            "Imagini_Laborator/lena512.bmp",
            "xray.png"
        };
        
        for (const auto& path : altPaths) {
            image = imread(path);
            if (!image.empty()) {
                imagePath = path;
                break;
            }
        }
        
        if (image.empty()) {
            cerr << "Eroare: Nu s-a putut încărca imaginea!\n";
            cerr << "Utilizare: ./lab11 <cale_imagine>\n";
            cerr << "\nPuteți folosi imaginea coins.png din lab10:\n";
            cerr << "  ./lab11 ../lab10/Imagini_Laborator/coins.png\n";
            return -1;
        }
    }
    
    cout << "\nImagine încărcată: " << imagePath << endl;
    printImageStats(image, imagePath);
    
    int option;
    do {
        displayMenu();
        cin >> option;
        
        switch (option) {
            case 1: exercitiul1(image); break;
            case 2: exercitiul2(image); break;
            case 3: exercitiul3(image); break;
            case 4: exercitiul4(image); break;
            case 5: exercitiul5(image); break;
            case 6: exercitiul6(image); break;
            case 7: runAllExercises(image); break;
            case 0: cout << "La revedere!\n"; break;
            default: cout << "Opțiune invalidă!\n";
        }
        
    } while (option != 0);
    
    destroyAllWindows();
    return 0;
}
