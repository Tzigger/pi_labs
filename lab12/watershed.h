#ifndef WATERSHED_H
#define WATERSHED_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

/**
 * PASUL 1: Segmentare grosieră cu threshold Otsu
 * Returnează imaginea binarizată
 */
Mat step1_otsuThreshold(const Mat& grayImage);

/**
 * PASUL 2: Corectarea erorilor de segmentare
 * - Închidere morfologică (elimină cavități interioare)
 * - Deschidere morfologică (elimină puncte exterioare)
 */
Mat step2_morphologicalCorrection(const Mat& binaryImage, int kernelSize = 5);

/**
 * PASUL 3: Identificarea background-ului cert
 * Dilatare morfologică pentru exagerarea contururilor
 * Returnează masca unde negru = background cert
 */
Mat step3_identifySureBackground(const Mat& correctedImage, int dilationSize = 3);

/**
 * PASUL 4: Calcularea transformatei distanță
 * Aplică distanceTransform cu DIST_L2 pe negativul imaginii
 */
Mat step4_distanceTransform(const Mat& correctedImage);

/**
 * PASUL 5: Identificarea interioarelor obiectelor (foreground cert)
 * Segmentare cu prag pe transformata distanță (ex: 80% din max)
 */
Mat step5_identifySureForeground(const Mat& distTransform, double threshold = 0.8);

/**
 * PASUL 6: Separarea regiunilor pe obiecte cu connectedComponents
 * Returnează matricea de label-uri (0 = background, 1,2,3... = obiecte)
 */
Mat step6_connectedComponents(const Mat& sureForeground, int& numLabels);

/**
 * PASUL 7: Identificarea regiunilor de incertitudine
 * Elimină foreground-ul cert din background-ul dilatat
 */
Mat step7_uncertainRegion(const Mat& sureBackground, const Mat& sureForeground);

/**
 * PASUL 8: Construirea matricei de markeri pentru watershed
 * - Background = 1
 * - Centre obiecte = 2, 3, 4, ...
 * - Regiuni incerte = 0
 */
Mat step8_buildMarkers(const Mat& labels, const Mat& uncertainRegion);

/**
 * PASUL 9: Aplicarea algoritmului watershed
 * Returnează imaginea cu contururile marcate
 */
Mat step9_applyWatershed(const Mat& originalImage, Mat& markers);

/**
 * Funcție auxiliară: Normalizează și afișează o matrice pentru vizualizare
 */
Mat normalizeForDisplay(const Mat& input);

/**
 * Funcție auxiliară: Vizualizare markeri în culori
 * Regiuni incerte = albastru, centre obiecte = culori random
 */
Mat visualizeMarkers(const Mat& markers, const Mat& uncertainRegion);

/**
 * Funcție completă care rulează toți pașii și salvează rezultatele intermediare
 */
void watershedComplete(const Mat& originalImage, const string& outputDir);

#endif // WATERSHED_H
