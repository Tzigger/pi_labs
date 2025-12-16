#include "watershed.h"
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

/**
 * PASUL 1: Segmentare grosieră cu threshold Otsu
 */
Mat step1_otsuThreshold(const Mat& grayImage) {
    Mat binary;
    // Aplicăm Otsu thresholding - THRESH_BINARY_INV pentru obiecte albe pe fundal negru
    double threshValue = threshold(grayImage, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    cout << "  Prag Otsu calculat: " << threshValue << endl;
    return binary;
}

/**
 * PASUL 2: Corectarea erorilor de segmentare
 * - Închidere morfologică (elimină cavități interioare)
 * - Deschidere morfologică (elimină puncte exterioare)
 */
Mat step2_morphologicalCorrection(const Mat& binaryImage, int kernelSize) {
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    Mat closed, opened;
    
    // Închidere: elimină cavitățile interioare (dilate apoi erode)
    morphologyEx(binaryImage, closed, MORPH_CLOSE, kernel);
    
    // Deschidere: elimină punctele exterioare (erode apoi dilate)
    morphologyEx(closed, opened, MORPH_OPEN, kernel);
    
    cout << "  Kernel morfologic: " << kernelSize << "x" << kernelSize << endl;
    return opened;
}

/**
 * PASUL 3: Identificarea background-ului cert
 * Dilatare morfologică pentru exagerarea contururilor
 */
Mat step3_identifySureBackground(const Mat& correctedImage, int dilationSize) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(dilationSize, dilationSize));
    Mat dilated;
    
    // Dilatăm imaginea corectată
    dilate(correctedImage, dilated, kernel);
    
    cout << "  Element structural: cerc " << dilationSize << "x" << dilationSize << endl;
    return dilated;
}

/**
 * PASUL 4: Calcularea transformatei distanță
 */
Mat step4_distanceTransform(const Mat& correctedImage) {
    Mat distTransform;
    
    // Aplicăm distanceTransform pe imaginea corectată (nu pe negativul ei)
    // OpenCV consideră automat pixelii nenuli ca fiind foreground
    distanceTransform(correctedImage, distTransform, DIST_L2, 5);
    
    double minVal, maxVal;
    minMaxLoc(distTransform, &minVal, &maxVal);
    cout << "  Distanță min: " << minVal << ", max: " << maxVal << endl;
    
    return distTransform;
}

/**
 * PASUL 5: Identificarea interioarelor obiectelor (foreground cert)
 */
Mat step5_identifySureForeground(const Mat& distTransform, double thresholdRatio) {
    double minVal, maxVal;
    minMaxLoc(distTransform, &minVal, &maxVal);
    
    // Pragul este un procentaj din valoarea maximă
    double threshValue = maxVal * thresholdRatio;
    
    Mat sureFg;
    threshold(distTransform, sureFg, threshValue, 255, THRESH_BINARY);
    
    // Convertim la CV_8U
    sureFg.convertTo(sureFg, CV_8U);
    
    cout << "  Prag distanță: " << threshValue << " (" << (thresholdRatio*100) << "% din max)" << endl;
    return sureFg;
}

/**
 * PASUL 6: Separarea regiunilor pe obiecte cu connectedComponents
 */
Mat step6_connectedComponents(const Mat& sureForeground, int& numLabels) {
    Mat labels;
    
    // Identificăm componentele conexe
    numLabels = connectedComponents(sureForeground, labels);
    
    cout << "  Componente conexe găsite: " << (numLabels - 1) << " (+ background)" << endl;
    return labels;
}

/**
 * PASUL 7: Identificarea regiunilor de incertitudine
 */
Mat step7_uncertainRegion(const Mat& sureBackground, const Mat& sureForeground) {
    Mat uncertain;
    
    // Regiunile incerte = Background dilatat - Foreground cert
    subtract(sureBackground, sureForeground, uncertain);
    
    int uncertainPixels = countNonZero(uncertain);
    cout << "  Pixeli în regiunea de incertitudine: " << uncertainPixels << endl;
    
    return uncertain;
}

/**
 * PASUL 8: Construirea matricei de markeri pentru watershed
 */
Mat step8_buildMarkers(const Mat& labels, const Mat& uncertainRegion) {
    Mat markers;
    
    // Adunăm 1 la toate label-urile
    // Background (0) devine 1, obiectele (1,2,3...) devin (2,3,4...)
    labels.convertTo(markers, CV_32S);
    markers = markers + 1;
    
    // Setăm pe 0 regiunile de incertitudine
    markers.setTo(0, uncertainRegion > 0);
    
    cout << "  Markeri construiți: background=1, obiecte=2+, incertitudine=0" << endl;
    return markers;
}

/**
 * PASUL 9: Aplicarea algoritmului watershed
 */
Mat step9_applyWatershed(const Mat& originalImage, Mat& markers) {
    Mat imageRGB;
    
    // Convertim imaginea în RGB (watershed necesită 3 canale)
    if (originalImage.channels() == 1) {
        cvtColor(originalImage, imageRGB, COLOR_GRAY2BGR);
    } else {
        imageRGB = originalImage.clone();
    }
    
    // Aplicăm watershed
    watershed(imageRGB, markers);
    
    // Creăm imaginea rezultat cu contururile marcate
    Mat result = imageRGB.clone();
    
    // Marcăm contururile (pixelii cu valoarea -1) în roșu
    result.setTo(Scalar(0, 0, 255), markers == -1);
    
    // Numărăm pixelii de contur
    int boundaryPixels = countNonZero(markers == -1);
    cout << "  Pixeli de contur (boundaries): " << boundaryPixels << endl;
    
    return result;
}

/**
 * Funcție auxiliară: Normalizează și afișează o matrice
 */
Mat normalizeForDisplay(const Mat& input) {
    Mat normalized;
    
    if (input.type() == CV_32S) {
        // Pentru matrici de întregi (labels, markers)
        double minVal, maxVal;
        minMaxLoc(input, &minVal, &maxVal);
        
        input.convertTo(normalized, CV_8U, 255.0 / maxVal);
    } else if (input.type() == CV_32F || input.type() == CV_64F) {
        // Pentru matrici float (distance transform)
        normalize(input, normalized, 0, 255, NORM_MINMAX, CV_8U);
    } else {
        // Deja CV_8U
        normalized = input.clone();
    }
    
    return normalized;
}

/**
 * Funcție auxiliară: Vizualizare markeri în culori
 * Regiuni incerte = albastru, centre obiecte = culori diferite
 */
Mat visualizeMarkers(const Mat& markers, const Mat& uncertainRegion) {
    Mat colorMarkers = Mat::zeros(markers.size(), CV_8UC3);
    
    // Generăm culori random pentru fiecare marker (obiect)
    vector<Vec3b> colors;
    colors.push_back(Vec3b(0, 0, 0));        // 0 = negru (incertitudine)
    colors.push_back(Vec3b(50, 50, 50));     // 1 = gri închis (background)
    
    // Culori pentru obiecte (2, 3, 4, ...)
    RNG rng(12345);
    for (int i = 2; i < 100; i++) {
        colors.push_back(Vec3b(rng.uniform(100, 255), 
                               rng.uniform(100, 255), 
                               rng.uniform(100, 255)));
    }
    
    // Colorăm fiecare pixel în funcție de valoarea marker-ului
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int label = markers.at<int>(i, j);
            if (label >= 0 && label < (int)colors.size()) {
                colorMarkers.at<Vec3b>(i, j) = colors[label];
            }
        }
    }
    
    // Marcăm regiunile de incertitudine în ALBASTRU
    colorMarkers.setTo(Scalar(255, 0, 0), uncertainRegion > 0);  // BGR: albastru
    
    return colorMarkers;
}

/**
 * Funcție completă care rulează toți pașii
 */
void watershedComplete(const Mat& originalImage, const string& outputDir) {
    // Verificăm dacă directorul de output există
    fs::create_directories(outputDir);
    
    cout << "\n========================================" << endl;
    cout << "  ALGORITMUL WATERSHED - 9 PAȘI" << endl;
    cout << "========================================\n" << endl;
    
    // Convertim în grayscale dacă e necesar
    Mat grayImage;
    if (originalImage.channels() == 3) {
        cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = originalImage.clone();
    }
    
    // PASUL 1: Segmentare Otsu
    cout << "PASUL 1: Segmentare cu threshold Otsu" << endl;
    Mat binary = step1_otsuThreshold(grayImage);
    imwrite(outputDir + "/step1_otsu_threshold.png", binary);
    
    // PASUL 2: Corectare morfologică
    cout << "\nPASUL 2: Corectare morfologică (închidere + deschidere)" << endl;
    Mat corrected = step2_morphologicalCorrection(binary, 3);
    imwrite(outputDir + "/step2_morphological_correction.png", corrected);
    
    // PASUL 3: Background cert
    cout << "\nPASUL 3: Identificare background cert (dilatare)" << endl;
    Mat sureBg = step3_identifySureBackground(corrected, 3);
    imwrite(outputDir + "/step3_sure_background.png", sureBg);
    
    // PASUL 4: Transformata distanță
    cout << "\nPASUL 4: Calculare transformată distanță" << endl;
    Mat distTransform = step4_distanceTransform(corrected);
    Mat distDisplay = normalizeForDisplay(distTransform);
    imwrite(outputDir + "/step4_distance_transform.png", distDisplay);
    
    // PASUL 5: Foreground cert
    cout << "\nPASUL 5: Identificare foreground cert (threshold pe distanță)" << endl;
    Mat sureFg = step5_identifySureForeground(distTransform, 0.7);
    imwrite(outputDir + "/step5_sure_foreground.png", sureFg);
    
    // PASUL 6: Connected components
    cout << "\nPASUL 6: Separare obiecte cu connectedComponents" << endl;
    int numLabels;
    Mat labels = step6_connectedComponents(sureFg, numLabels);
    Mat labelsDisplay = normalizeForDisplay(labels);
    imwrite(outputDir + "/step6_connected_components.png", labelsDisplay);
    
    // PASUL 7: Regiunea de incertitudine
    cout << "\nPASUL 7: Identificare regiuni de incertitudine" << endl;
    Mat uncertain = step7_uncertainRegion(sureBg, sureFg);
    imwrite(outputDir + "/step7_uncertain_region.png", uncertain);
    
    // PASUL 8: Construire markeri
    cout << "\nPASUL 8: Construire matrice de markeri" << endl;
    Mat markers = step8_buildMarkers(labels, uncertain);
    Mat markersDisplay = normalizeForDisplay(markers);
    imwrite(outputDir + "/step8_markers.png", markersDisplay);
    
    // Salvăm și vizualizarea colorată cu regiunile de incertitudine în albastru
    Mat markersColor = visualizeMarkers(markers, uncertain);
    imwrite(outputDir + "/step8_markers_color.png", markersColor);
    
    // PASUL 9: Watershed
    cout << "\nPASUL 9: Aplicare algoritm watershed" << endl;
    Mat result = step9_applyWatershed(originalImage, markers);
    imwrite(outputDir + "/step9_watershed_result.png", result);
    
    // Salvăm și markerii finali
    Mat markersFinal = normalizeForDisplay(markers);
    imwrite(outputDir + "/step9_markers_final.png", markersFinal);
    
    cout << "\n========================================" << endl;
    cout << "  TOATE PASURILE COMPLETATE!" << endl;
    cout << "  Output salvat în: " << outputDir << endl;
    cout << "========================================\n" << endl;
}
