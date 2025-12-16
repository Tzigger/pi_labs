#include "watershed.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Variabile globale pentru imagine
Mat g_originalImage;
Mat g_grayImage;
string g_outputDir = "output_lab12";

void printMenu() {
    cout << "\n====== LABORATOR 12 - Watershed ======" << endl;
    cout << "1. Pasul 1 - Segmentare Otsu" << endl;
    cout << "2. Pasul 2 - Corectare morfologică" << endl;
    cout << "3. Pasul 3 - Identificare background cert" << endl;
    cout << "4. Pasul 4 - Transformată distanță" << endl;
    cout << "5. Pasul 5 - Identificare foreground cert" << endl;
    cout << "6. Pasul 6 - Connected components" << endl;
    cout << "7. Pasul 7 - Regiuni de incertitudine" << endl;
    cout << "8. Pasul 8 - Construire markeri" << endl;
    cout << "9. Pasul 9 - Aplicare watershed" << endl;
    cout << "10. Rulează TOȚI pașii (1-9) și salvează" << endl;
    cout << "0. Ieșire" << endl;
    cout << "======================================" << endl;
    cout << "Alegeți opțiunea: ";
}

void exercitiul1() {
    cout << "\n=== PASUL 1: Segmentare cu threshold Otsu ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    
    imshow("Original", g_grayImage);
    imshow("Pasul 1 - Otsu Threshold", binary);
    
    string filename = g_outputDir + "/step1_otsu_threshold.png";
    imwrite(filename, binary);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul2() {
    cout << "\n=== PASUL 2: Corectare morfologică ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    
    imshow("Pasul 1 - Otsu", binary);
    imshow("Pasul 2 - Corectare morfologică", corrected);
    
    string filename = g_outputDir + "/step2_morphological_correction.png";
    imwrite(filename, corrected);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul3() {
    cout << "\n=== PASUL 3: Identificare background cert ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat sureBg = step3_identifySureBackground(corrected, 3);
    
    imshow("Pasul 2 - Corectare", corrected);
    imshow("Pasul 3 - Background cert (dilatat)", sureBg);
    
    string filename = g_outputDir + "/step3_sure_background.png";
    imwrite(filename, sureBg);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul4() {
    cout << "\n=== PASUL 4: Transformată distanță ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    
    Mat distDisplay = normalizeForDisplay(distTransform);
    
    imshow("Pasul 2 - Corectare", corrected);
    imshow("Pasul 4 - Transformată distanță", distDisplay);
    
    string filename = g_outputDir + "/step4_distance_transform.png";
    imwrite(filename, distDisplay);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul5() {
    cout << "\n=== PASUL 5: Identificare foreground cert ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    
    Mat distDisplay = normalizeForDisplay(distTransform);
    
    imshow("Pasul 4 - Transformată distanță", distDisplay);
    imshow("Pasul 5 - Foreground cert", sureFg);
    
    string filename = g_outputDir + "/step5_sure_foreground.png";
    imwrite(filename, sureFg);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul6() {
    cout << "\n=== PASUL 6: Connected components ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    
    int numLabels;
    Mat labels = step6_connectedComponents(sureFg, numLabels);
    Mat labelsDisplay = normalizeForDisplay(labels);
    
    imshow("Pasul 5 - Foreground cert", sureFg);
    imshow("Pasul 6 - Connected components", labelsDisplay);
    
    string filename = g_outputDir + "/step6_connected_components.png";
    imwrite(filename, labelsDisplay);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul7() {
    cout << "\n=== PASUL 7: Regiuni de incertitudine ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat sureBg = step3_identifySureBackground(corrected, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    Mat uncertain = step7_uncertainRegion(sureBg, sureFg);
    
    imshow("Pasul 3 - Background cert", sureBg);
    imshow("Pasul 5 - Foreground cert", sureFg);
    imshow("Pasul 7 - Regiuni incerte", uncertain);
    
    string filename = g_outputDir + "/step7_uncertain_region.png";
    imwrite(filename, uncertain);
    cout << "✓ Salvat: " << filename << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul8() {
    cout << "\n=== PASUL 8: Construire markeri ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat sureBg = step3_identifySureBackground(corrected, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    
    int numLabels;
    Mat labels = step6_connectedComponents(sureFg, numLabels);
    Mat uncertain = step7_uncertainRegion(sureBg, sureFg);
    Mat markers = step8_buildMarkers(labels, uncertain);
    
    Mat markersDisplay = normalizeForDisplay(markers);
    Mat markersColor = visualizeMarkers(markers, uncertain);
    Mat labelsDisplay = normalizeForDisplay(labels);
    
    imshow("Pasul 6 - Labels", labelsDisplay);
    imshow("Pasul 7 - Incertitudine", uncertain);
    imshow("Pasul 8 - Markeri (grayscale)", markersDisplay);
    imshow("Pasul 8 - Markeri (color - albastru=incertitudine)", markersColor);
    
    string filename1 = g_outputDir + "/step8_markers.png";
    string filename2 = g_outputDir + "/step8_markers_color.png";
    imwrite(filename1, markersDisplay);
    imwrite(filename2, markersColor);
    cout << "✓ Salvat: " << filename1 << endl;
    cout << "✓ Salvat: " << filename2 << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul9() {
    cout << "\n=== PASUL 9: Aplicare watershed ===" << endl;
    
    Mat binary = step1_otsuThreshold(g_grayImage);
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    Mat sureBg = step3_identifySureBackground(corrected, 3);
    Mat distTransform = step4_distanceTransform(corrected);
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    
    int numLabels;
    Mat labels = step6_connectedComponents(sureFg, numLabels);
    Mat uncertain = step7_uncertainRegion(sureBg, sureFg);
    Mat markers = step8_buildMarkers(labels, uncertain);
    
    Mat result = step9_applyWatershed(g_originalImage, markers);
    
    Mat markersDisplay = normalizeForDisplay(markers);
    
    imshow("Original", g_originalImage);
    imshow("Pasul 8 - Markeri", markersDisplay);
    imshow("Pasul 9 - Watershed Result", result);
    
    string filename1 = g_outputDir + "/step9_watershed_result.png";
    string filename2 = g_outputDir + "/step9_markers_final.png";
    imwrite(filename1, result);
    imwrite(filename2, markersDisplay);
    cout << "✓ Salvat: " << filename1 << endl;
    cout << "✓ Salvat: " << filename2 << endl;
    
    waitKey(0);
    destroyAllWindows();
}

void exercitiul10() {
    cout << "\n========================================" << endl;
    cout << "  RULARE TOȚI PAȘII (1-9)" << endl;
    cout << "========================================\n" << endl;
    
    watershedComplete(g_originalImage, g_outputDir);
    
    cout << "\n========================================" << endl;
    cout << "  TOATE REZULTATELE AU FOST SALVATE!" << endl;
    cout << "  Verificați directorul: " << g_outputDir << "/" << endl;
    cout << "========================================\n" << endl;
}

int main(int argc, char** argv) {
    cout << "============================================" << endl;
    cout << "   LABORATOR 12 - Watershed Segmentation   " << endl;
    cout << "============================================\n" << endl;
    
    // Încărcăm imaginea
    string imagePath;
    if (argc > 1) {
        imagePath = argv[1];
    } else {
        cout << "Utilizare: " << argv[0] << " <cale_imagine>" << endl;
        cout << "Exemplu: " << argv[0] << " Imagini_Laborator/coins.png" << endl;
        return -1;
    }
    
    // Citim imaginea
    g_originalImage = imread(imagePath);
    if (g_originalImage.empty()) {
        cerr << "Eroare: Nu se poate încărca imaginea: " << imagePath << endl;
        return -1;
    }
    
    cout << "Imagine încărcată: " << imagePath << endl;
    
    // Convertim în grayscale
    if (g_originalImage.channels() == 3) {
        cvtColor(g_originalImage, g_grayImage, COLOR_BGR2GRAY);
    } else {
        g_grayImage = g_originalImage.clone();
    }
    
    // Afișăm statistici
    cout << "=== Statistici pentru " << imagePath << " ===" << endl;
    cout << "Dimensiuni: " << g_originalImage.cols << " x " << g_originalImage.rows << endl;
    cout << "Canale: " << g_originalImage.channels() << endl;
    
    // Creăm directorul de output
    fs::create_directories(g_outputDir);
    
    // Meniu interactiv
    int choice;
    do {
        printMenu();
        cin >> choice;
        
        switch(choice) {
            case 1: exercitiul1(); break;
            case 2: exercitiul2(); break;
            case 3: exercitiul3(); break;
            case 4: exercitiul4(); break;
            case 5: exercitiul5(); break;
            case 6: exercitiul6(); break;
            case 7: exercitiul7(); break;
            case 8: exercitiul8(); break;
            case 9: exercitiul9(); break;
            case 10: exercitiul10(); break;
            case 0: cout << "La revedere!" << endl; break;
            default: cout << "Opțiune invalidă!" << endl;
        }
    } while (choice != 0);
    
    return 0;
}
