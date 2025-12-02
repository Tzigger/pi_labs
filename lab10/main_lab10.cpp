/**
 * Laborator 10 - Segmentarea Imaginilor (1)
 * 
 * Exerciții:
 * 1. Determinarea pragului de segmentare (Otsu și metoda iterativă)
 * 2. Afișarea imaginii inițiale și rezultatul segmentării
 * 3. Afișarea componentelor etichetate (connectedComponents)
 * 4. Identificarea obiectelor individuale + operații morfologice
 * 5. Izolarea obiectelor (bounding box)
 * 6. Segmentare cu prag local pentru fiecare obiect
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include "segmentation_helpers.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Variabile globale pentru imagine
Mat g_originalImage;
Mat g_grayImage;
Mat g_binaryImage;
Mat g_labeledImage;
int g_numLabels = 0;

void displayMenu() {
    cout << "\n====== LABORATOR 10 - Segmentarea Imaginilor (1) ======\n";
    cout << "1. Determinare prag segmentare (Otsu + Iterativ)\n";
    cout << "2. Afișare imagine inițială și rezultat segmentare\n";
    cout << "3. Afișare componente etichetate (connectedComponents)\n";
    cout << "4. Identificare obiecte + Operații morfologice\n";
    cout << "5. Izolare obiecte (bounding box)\n";
    cout << "6. Segmentare cu prag local pentru fiecare obiect\n";
    cout << "7. Rulează TOATE exercițiile și salvează rezultatele\n";
    cout << "0. Ieșire\n";
    cout << "========================================================\n";
    cout << "Alegeți opțiunea: ";
}

// =============================================================================
// EXERCIȚIUL 1: Determinarea pragului de segmentare
// =============================================================================
void exercitiul1(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 1: Determinarea pragului de segmentare ===\n";
    
    Mat gray = toGrayscale(image);
    
    // 1a. Metoda Otsu folosind cv::threshold
    Mat binaryOtsu;
    double otsuThreshold = threshold(gray, binaryOtsu, 0, 255, THRESH_BINARY | THRESH_OTSU);
    cout << "1a. Pragul Otsu (cv::threshold): " << otsuThreshold << endl;
    
    // 1b. Metoda iterativă
    // Inițializare: media între min și max (pentru obiecte mici pe fundal mare)
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    double T = (minVal + maxVal) / 2.0;
    
    double Tprev;
    int iteration = 0;
    const double epsilon = 0.5;
    const int maxIterations = 100;
    
    cout << "\n1b. Metoda iterativă:\n";
    cout << "    Valoare inițială T = " << T << " (mediana min-max)\n";
    
    do {
        Tprev = T;
        
        double sum1 = 0, count1 = 0;  // pixeli < T
        double sum2 = 0, count2 = 0;  // pixeli >= T
        
        for (int y = 0; y < gray.rows; y++) {
            for (int x = 0; x < gray.cols; x++) {
                uchar pixel = gray.at<uchar>(y, x);
                if (pixel < T) {
                    sum1 += pixel;
                    count1++;
                } else {
                    sum2 += pixel;
                    count2++;
                }
            }
        }
        
        double m1 = (count1 > 0) ? sum1 / count1 : 0;
        double m2 = (count2 > 0) ? sum2 / count2 : 255;
        
        T = (m1 + m2) / 2.0;
        iteration++;
        
        cout << "    Iterația " << iteration << ": T = " << T 
             << " (m1=" << m1 << ", m2=" << m2 << ")\n";
        
    } while (abs(T - Tprev) > epsilon && iteration < maxIterations);
    
    int iterativeThreshold = (int)round(T);
    cout << "    Prag iterativ final: " << iterativeThreshold << " (după " << iteration << " iterații)\n";
    
    // Aplicăm pragul iterativ
    Mat binaryIterative;
    threshold(gray, binaryIterative, iterativeThreshold, 255, THRESH_BINARY);
    
    // Salvăm rezultatele
    imwrite("output_lab10/ex1_original_gray.png", gray);
    imwrite("output_lab10/ex1_otsu_T" + to_string((int)otsuThreshold) + ".png", binaryOtsu);
    imwrite("output_lab10/ex1_iterativ_T" + to_string(iterativeThreshold) + ".png", binaryIterative);
    
    cout << "\n✓ Salvat: output_lab10/ex1_original_gray.png\n";
    cout << "✓ Salvat: output_lab10/ex1_otsu_T" << (int)otsuThreshold << ".png\n";
    cout << "✓ Salvat: output_lab10/ex1_iterativ_T" << iterativeThreshold << ".png\n";
    
    // Actualizăm variabilele globale
    g_grayImage = gray;
    g_binaryImage = binaryOtsu.clone();
    
    if (!saveOnly) {
        // Afișăm comparația
        Mat comparison;
        vector<Mat> images = {gray, binaryOtsu, binaryIterative};
        
        // Convertim la color pentru afișare cu text
        for (auto& img : images) {
            if (img.channels() == 1) {
                cvtColor(img, img, COLOR_GRAY2BGR);
            }
        }
        
        hconcat(images, comparison);
        
        // Adăugăm text
        putText(comparison, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        putText(comparison, "Otsu T=" + to_string((int)otsuThreshold), 
                Point(gray.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        putText(comparison, "Iterativ T=" + to_string(iterativeThreshold), 
                Point(2 * gray.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        
        imwrite("output_lab10/ex1_comparison.png", comparison);
        imshow("Exercitiul 1 - Determinare Prag", comparison);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 2: Afișarea imaginii inițiale și rezultatul segmentării
// =============================================================================
void exercitiul2(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 2: Afișare imagine inițială și segmentare ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Segmentare cu Otsu
    Mat binary;
    double T = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    cout << "Prag utilizat: " << T << endl;
    
    // Inversăm imaginea binară (obiecte negre pe fundal alb -> obiecte albe pe fundal negru)
    Mat binaryInv;
    bitwise_not(binary, binaryInv);
    
    // Creăm afișarea side-by-side
    Mat grayColor, binaryColor;
    cvtColor(gray, grayColor, COLOR_GRAY2BGR);
    cvtColor(binaryInv, binaryColor, COLOR_GRAY2BGR);
    
    Mat display;
    hconcat(grayColor, binaryColor, display);
    
    // Adăugăm text
    putText(display, "Imaginea initiala", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(display, "Segmentare (T=" + to_string((int)T) + ")", 
            Point(gray.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    
    imwrite("output_lab10/ex2_initial_vs_segmentat.png", display);
    cout << "✓ Salvat: output_lab10/ex2_initial_vs_segmentat.png\n";
    
    // Actualizăm variabilele globale (folosim imaginea inversată)
    g_grayImage = gray;
    g_binaryImage = binaryInv;
    
    if (!saveOnly) {
        imshow("Exercitiul 2 - Initial vs Segmentat", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 3: Afișare componente etichetate (connectedComponents)
// =============================================================================
void exercitiul3(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 3: Componente etichetate (connectedComponents) ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Binarizăm imaginea - monedele sunt mai luminoase, deci THRESH_BINARY_INV
    // face monedele ALBE (pentru connectedComponents să le detecteze)
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    // Aplicăm connectedComponents pe imaginea binarizată FĂRĂ closing
    // pentru a vedea toate componentele (inclusiv găurile din monede)
    Mat labels;
    g_numLabels = connectedComponents(binary, labels, 8, CV_32S);
    
    cout << "Număr de componente găsite: " << g_numLabels - 1 << " (excluzând fundalul)\n";
    
    // Normalizăm etichetele la interval 0-255 pentru vizualizare
    // Fiecare componentă primește o nuanță diferită de gri
    Mat labelsGray = Mat::zeros(labels.size(), CV_8UC1);
    
    // Normalizăm corect: label 0 = fundal (negru), restul = tonuri de gri
    double maxLabel = g_numLabels - 1;
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                // Normalizăm la intervalul 50-255 pentru vizibilitate
                labelsGray.at<uchar>(y, x) = (uchar)(50 + (label * 205.0 / maxLabel));
            }
        }
    }
    
    // Salvăm variabila globală
    g_labeledImage = labels.clone();
    g_binaryImage = binary.clone();
    
    // Afișare: imaginea binarizată și componentele în tonuri de gri
    Mat binaryDisplay, labelsDisplay;
    cvtColor(binary, binaryDisplay, COLOR_GRAY2BGR);
    cvtColor(labelsGray, labelsDisplay, COLOR_GRAY2BGR);
    
    Mat display;
    hconcat(binaryDisplay, labelsDisplay, display);
    
    putText(display, "Imagine binarizata", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    putText(display, "Componente etichetate (" + to_string(g_numLabels - 1) + " obiecte)", 
            Point(binary.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    
    imwrite("output_lab10/ex3_labeled_grayscale.png", labelsGray);
    imwrite("output_lab10/ex3_binary.png", binary);
    imwrite("output_lab10/ex3_comparison.png", display);
    
    cout << "✓ Salvat: output_lab10/ex3_labeled_grayscale.png\n";
    cout << "✓ Salvat: output_lab10/ex3_binary.png\n";
    cout << "✓ Salvat: output_lab10/ex3_comparison.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 3 - Componente Etichetate", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 4: Identificare obiecte + Operații morfologice
// =============================================================================
void exercitiul4(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 4: Obiecte individuale + Operații morfologice ===\n";
    
    Mat gray = toGrayscale(image);
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    
    // Găsim componentele conectate
    Mat labels;
    int numLabels = connectedComponents(binary, labels, 8, CV_32S);
    
    cout << "Număr componente găsite: " << numLabels - 1 << endl;
    cout << "Afișare obiecte individuale la poziția lor originală...\n";
    
    int objectsToShow = min(4, numLabels - 1);
    
    for (int label = 1; label <= objectsToShow; label++) {
        // Creăm masca pentru acest obiect (imagine full-size, doar obiectul alb)
        Mat maskBinary = Mat::zeros(binary.size(), CV_8UC1);
        Mat maskGray = Mat::zeros(gray.size(), CV_8UC1);
        
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                if (labels.at<int>(y, x) == label) {
                    maskBinary.at<uchar>(y, x) = 255;  // Obiectul segmentat (alb)
                    maskGray.at<uchar>(y, x) = gray.at<uchar>(y, x);  // Imaginea originală
                }
            }
        }
        
        // Salvăm ambele variante: segmentată și originală
        imwrite("output_lab10/ex4_obiect_" + to_string(label) + "_segmentat.png", maskBinary);
        imwrite("output_lab10/ex4_obiect_" + to_string(label) + "_original.png", maskGray);
        
        cout << "✓ Salvat obiect " << label << " (segmentat + original)\n";
    }
    
    // Operații morfologice pentru corectarea erorilor
    cout << "\nAplicare operații morfologice...\n";
    
    // Element structural eliptic
    Mat elementEllipse = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    // Element structural dreptunghiular
    Mat elementRect = getStructuringElement(MORPH_RECT, Size(10, 10));
    
    Mat dilated, eroded, opened, closed;
    
    // Dilatare
    morphologyEx(binary, dilated, MORPH_DILATE, elementEllipse);
    imwrite("output_lab10/ex4_morph_dilate.png", dilated);
    
    // Eroziune
    morphologyEx(binary, eroded, MORPH_ERODE, elementEllipse);
    imwrite("output_lab10/ex4_morph_erode.png", eroded);
    
    // Deschidere (eroziune + dilatare) - elimină zgomot
    morphologyEx(binary, opened, MORPH_OPEN, elementEllipse);
    imwrite("output_lab10/ex4_morph_open.png", opened);
    
    // Închidere (dilatare + eroziune) - umple găuri
    morphologyEx(binary, closed, MORPH_CLOSE, elementEllipse);
    imwrite("output_lab10/ex4_morph_close.png", closed);
    
    cout << "✓ Salvat: output_lab10/ex4_morph_dilate.png\n";
    cout << "✓ Salvat: output_lab10/ex4_morph_erode.png\n";
    cout << "✓ Salvat: output_lab10/ex4_morph_open.png\n";
    cout << "✓ Salvat: output_lab10/ex4_morph_close.png\n";
    
    // Creăm comparația operațiilor morfologice
    Mat row1, row2, comparison;
    
    Mat binaryColor, dilatedColor, erodedColor, openedColor, closedColor;
    cvtColor(binary, binaryColor, COLOR_GRAY2BGR);
    cvtColor(dilated, dilatedColor, COLOR_GRAY2BGR);
    cvtColor(eroded, erodedColor, COLOR_GRAY2BGR);
    cvtColor(opened, openedColor, COLOR_GRAY2BGR);
    cvtColor(closed, closedColor, COLOR_GRAY2BGR);
    
    // Redimensionăm pentru afișare
    int targetWidth = 250;
    double scale = (double)targetWidth / binary.cols;
    Size newSize(targetWidth, (int)(binary.rows * scale));
    
    resize(binaryColor, binaryColor, newSize);
    resize(dilatedColor, dilatedColor, newSize);
    resize(erodedColor, erodedColor, newSize);
    resize(openedColor, openedColor, newSize);
    resize(closedColor, closedColor, newSize);
    
    putText(binaryColor, "Original", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(dilatedColor, "Dilatare", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(erodedColor, "Eroziune", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(openedColor, "Deschidere", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    putText(closedColor, "Inchidere", Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    
    hconcat(vector<Mat>{binaryColor, dilatedColor, erodedColor}, row1);
    hconcat(vector<Mat>{openedColor, closedColor}, row2);
    
    // Padding pentru row2
    Mat padding(row2.rows, row1.cols - row2.cols, CV_8UC3, Scalar(255, 255, 255));
    hconcat(row2, padding, row2);
    
    vconcat(row1, row2, comparison);
    
    imwrite("output_lab10/ex4_comparison.png", comparison);
    cout << "✓ Salvat: output_lab10/ex4_comparison.png\n";
    
    // Actualizăm imaginea binarizată cu versiunea corectată (closed)
    g_binaryImage = closed.clone();
    
    if (!saveOnly) {
        imshow("Exercitiul 4 - Operatii Morfologice", comparison);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 5: Izolare obiecte (bounding box)
// =============================================================================
void exercitiul5(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 5: Izolare obiecte (bounding box) ===\n";
    
    Mat gray = toGrayscale(image);
    Mat binaryTemp, binary;
    threshold(gray, binaryTemp, 0, 255, THRESH_BINARY | THRESH_OTSU);
    bitwise_not(binaryTemp, binary);  // Inversăm: obiecte albe pe fundal negru
    
    // Aplicăm closing pentru a umple găurile
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    Mat closed;
    morphologyEx(binary, closed, MORPH_CLOSE, element);
    
    // Găsim componentele conectate
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(closed, labels, stats, centroids, 8, CV_32S);
    
    cout << "Număr obiecte găsite: " << numLabels - 1 << endl;
    
    // Creăm imaginea cu bounding boxes
    Mat imageWithBoxes;
    cvtColor(gray, imageWithBoxes, COLOR_GRAY2BGR);
    
    vector<Mat> isolatedObjects;
    vector<Rect> boundingBoxes;
    
    for (int label = 1; label < numLabels; label++) {
        int x = stats.at<int>(label, CC_STAT_LEFT);
        int y = stats.at<int>(label, CC_STAT_TOP);
        int w = stats.at<int>(label, CC_STAT_WIDTH);
        int h = stats.at<int>(label, CC_STAT_HEIGHT);
        int area = stats.at<int>(label, CC_STAT_AREA);
        
        // Filtrăm obiectele prea mici
        if (area < 100) continue;
        
        Rect bbox(x, y, w, h);
        boundingBoxes.push_back(bbox);
        
        // Desenăm bounding box
        rectangle(imageWithBoxes, bbox, Scalar(0, 255, 0), 2);
        putText(imageWithBoxes, to_string(label), Point(x, y - 5), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        
        // Extragem obiectul izolat
        Mat isolated = gray(bbox).clone();
        isolatedObjects.push_back(isolated);
        
        imwrite("output_lab10/ex5_obiect_izolat_" + to_string(label) + ".png", isolated);
    }
    
    imwrite("output_lab10/ex5_bounding_boxes.png", imageWithBoxes);
    cout << "✓ Salvat: output_lab10/ex5_bounding_boxes.png\n";
    cout << "✓ Salvat " << isolatedObjects.size() << " obiecte izolate\n";
    
    // Creăm o imagine cu primele obiecte izolate
    if (!isolatedObjects.empty()) {
        int maxObjects = min(6, (int)isolatedObjects.size());
        vector<Mat> displayObjects;
        
        int targetSize = 80;
        for (int i = 0; i < maxObjects; i++) {
            Mat resized;
            double scale = (double)targetSize / max(isolatedObjects[i].cols, isolatedObjects[i].rows);
            resize(isolatedObjects[i], resized, Size(), scale, scale);
            
            // Padding pentru a face toate la aceeași dimensiune
            Mat padded(targetSize, targetSize, CV_8UC1, Scalar(255));
            int offsetX = (targetSize - resized.cols) / 2;
            int offsetY = (targetSize - resized.rows) / 2;
            resized.copyTo(padded(Rect(offsetX, offsetY, resized.cols, resized.rows)));
            
            displayObjects.push_back(padded);
        }
        
        Mat objectsRow;
        hconcat(displayObjects, objectsRow);
        imwrite("output_lab10/ex5_obiecte_izolate_grid.png", objectsRow);
        cout << "✓ Salvat: output_lab10/ex5_obiecte_izolate_grid.png\n";
    }
    
    if (!saveOnly) {
        imshow("Exercitiul 5 - Bounding Boxes", imageWithBoxes);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 6: Segmentare cu prag local pentru fiecare obiect
// =============================================================================
void exercitiul6(const Mat& image, bool saveOnly = false) {
    cout << "\n=== EXERCIȚIUL 6: Segmentare cu prag local ===\n";
    
    Mat gray = toGrayscale(image);
    
    // Segmentare globală pentru comparație
    Mat binaryGlobalTemp, binaryGlobal;
    double globalT = threshold(gray, binaryGlobalTemp, 0, 255, THRESH_BINARY | THRESH_OTSU);
    bitwise_not(binaryGlobalTemp, binaryGlobal);  // Inversăm: obiecte albe pe fundal negru
    cout << "Prag global Otsu: " << globalT << endl;
    
    // Aplicăm closing pentru a găsi obiectele
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
    Mat closed;
    morphologyEx(binaryGlobal, closed, MORPH_CLOSE, element);
    
    // Găsim componentele conectate
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(closed, labels, stats, centroids, 8, CV_32S);
    
    // Creăm imaginea cu segmentare locală
    Mat binaryLocal = Mat::zeros(gray.size(), CV_8UC1);
    
    cout << "\nPraguri locale pentru fiecare obiect:\n";
    
    for (int label = 1; label < numLabels; label++) {
        int x = stats.at<int>(label, CC_STAT_LEFT);
        int y = stats.at<int>(label, CC_STAT_TOP);
        int w = stats.at<int>(label, CC_STAT_WIDTH);
        int h = stats.at<int>(label, CC_STAT_HEIGHT);
        int area = stats.at<int>(label, CC_STAT_AREA);
        
        // Filtrăm obiectele prea mici
        if (area < 100) continue;
        
        Rect bbox(x, y, w, h);
        
        // Extragem regiunea
        Mat roi = gray(bbox);
        
        // Calculăm pragul local cu Otsu și inversăm
        Mat localBinaryTemp, localBinary;
        double localT = threshold(roi, localBinaryTemp, 0, 255, THRESH_BINARY | THRESH_OTSU);
        bitwise_not(localBinaryTemp, localBinary);  // Inversăm
        
        cout << "  Obiect " << label << ": prag local = " << localT 
             << " (diferență față de global: " << (localT - globalT) << ")\n";
        
        // Copiem rezultatul în imaginea finală
        localBinary.copyTo(binaryLocal(bbox));
        
        // Salvăm comparația pentru acest obiect
        Mat roiGlobalBinary = binaryGlobal(bbox);
        
        Mat comparison;
        Mat roiColor, localColor, globalColor;
        cvtColor(roi, roiColor, COLOR_GRAY2BGR);
        cvtColor(localBinary, localColor, COLOR_GRAY2BGR);
        cvtColor(roiGlobalBinary, globalColor, COLOR_GRAY2BGR);
        
        hconcat(vector<Mat>{roiColor, globalColor, localColor}, comparison);
        imwrite("output_lab10/ex6_obiect_" + to_string(label) + "_comparison.png", comparison);
    }
    
    // Creăm comparația globală vs locală
    Mat globalColor, localColor;
    cvtColor(binaryGlobal, globalColor, COLOR_GRAY2BGR);
    cvtColor(binaryLocal, localColor, COLOR_GRAY2BGR);
    
    Mat display;
    hconcat(globalColor, localColor, display);
    
    putText(display, "Segmentare GLOBALA (T=" + to_string((int)globalT) + ")", 
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(display, "Segmentare LOCALA (prag individual)", 
            Point(gray.cols + 10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    
    imwrite("output_lab10/ex6_global_vs_local.png", display);
    imwrite("output_lab10/ex6_segmentare_locala.png", binaryLocal);
    
    cout << "\n✓ Salvat: output_lab10/ex6_global_vs_local.png\n";
    cout << "✓ Salvat: output_lab10/ex6_segmentare_locala.png\n";
    
    if (!saveOnly) {
        imshow("Exercitiul 6 - Global vs Local", display);
        waitKey(0);
    }
}

// =============================================================================
// EXERCIȚIUL 7: Rulează TOATE exercițiile
// =============================================================================
void runAllExercises(const Mat& image) {
    cout << "\n========================================\n";
    cout << "  RULARE TOATE EXERCIȚIILE (1-6)\n";
    cout << "========================================\n";
    
    exercitiul1(image, true);
    exercitiul2(image, true);
    exercitiul3(image, true);
    exercitiul4(image, true);
    exercitiul5(image, true);
    exercitiul6(image, true);
    
    cout << "\n========================================\n";
    cout << "  TOATE REZULTATELE AU FOST SALVATE!\n";
    cout << "  Verificați directorul: output_lab10/\n";
    cout << "========================================\n";
}

int main(int argc, char** argv) {
    cout << "============================================\n";
    cout << "   LABORATOR 10 - Segmentarea Imaginilor   \n";
    cout << "============================================\n";
    
    // Creăm directorul de output
    fs::create_directories("output_lab10");
    
    // Încărcăm imaginea de test
    string imagePath = "Imagini_Laborator/coins.png";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    Mat image = imread(imagePath);
    if (image.empty()) {
        // Încercăm alte căi
        vector<string> altPaths = {
            "Imagini_Laborator/coins.png",
            "../lab9/Imagini_Laborator/coins.png",
            "coins.png"
        };
        
        for (const auto& path : altPaths) {
            image = imread(path);
            if (!image.empty()) {
                imagePath = path;
                break;
            }
        }
        
        if (image.empty()) {
            cerr << "Eroare: Nu s-a putut încărca nicio imagine!\n";
            cerr << "Utilizare: ./lab10 <cale_imagine>\n";
            return -1;
        }
    }
    
    g_originalImage = image.clone();
    
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
