#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include "frequency_domain_helpers.h"

using namespace cv;
using namespace std;

void displayMenu() {
    cout << "\n====== LABORATOR 9 - Prelucrarea imaginilor in domeniul frecventelor ======\n";
    cout << "1. Afiseaza imaginea si transformata Fourier\n";
    cout << "2. Filtrare in domeniu frecventa pornind de la filtru gaussian spatial\n";
    cout << "3. Filtrare in domeniul frecventa pornind de la filtru gaussian in domeniul frecventa\n";
    cout << "0. Iesire\n";
    cout << "===============================================================================\n";
    cout << "Alegeti optiunea: ";
}

// Punctul 1: Afișarea unei imagini și a transformatei Fourier
void showFourierTransform(const Mat& image, const string& windowName = "Laborator 9") {
    cout << "\n=== Punctul 1: Afișarea imaginii și transformatei Fourier ===";
    cout << "\nSe folosește funcția cv::dft pentru calcularea DFT\n";
    
    Mat complexImage;
    Mat spectrum = computeFourierSpectrum(image, complexImage);
    
    // Afișăm imaginea originală și spectrul
    Mat display;
    Mat imageDisplay;
    
    // Convertim imaginea la grayscale pentru afișare dacă este color
    if (image.channels() == 3) {
        cvtColor(image, imageDisplay, COLOR_BGR2GRAY);
    } else {
        imageDisplay = image.clone();
    }
    
    // Redimensionăm spectrul la dimensiunea imaginii originale pentru comparație
    Mat spectrumResized;
    resize(spectrum, spectrumResized, imageDisplay.size());
    
    hconcat(imageDisplay, spectrumResized, display);
    
    // Salvăm rezultatele
    imwrite("output_lab9/punctul1_original.png", imageDisplay);
    imwrite("output_lab9/punctul1_spectru_fourier.png", spectrumResized);
    imwrite("output_lab9/punctul1_combinat.png", display);
    
    cout << "Afișare: Imaginea originală | Spectrul Fourier\n";
    cout << "✓ Salvat: output_lab9/punctul1_original.png\n";
    cout << "✓ Salvat: output_lab9/punctul1_spectru_fourier.png\n";
    cout << "✓ Salvat: output_lab9/punctul1_combinat.png\n";
    
    imshow(windowName, display);
    cout << "Apasati orice tasta pentru a continua...\n";
    waitKey(0);
}

// Punctul 2: Filtrarea în domeniul frecvență pornind de la un filtru gaussian spatial
void filterFromSpatialGaussian(const Mat& image) {
    cout << "\n=== Punctul 2: Filtrare în domeniu frecvență pornind de la filtru gaussian spatial ===\n";
    
    // Pasul 1: Creăm filtrul gaussian spatial 5x5
    cout << "Pasul 1: Crearea filtrului gaussian spatial 5x5\n";
    Mat spatialFilter = (Mat_<float>(5, 5) <<
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1);
    
    // Normalizăm filtrul
    spatialFilter = spatialFilter / 256.0f;
    
    cout << "Filtru gaussian spatial creat (suma = " << sum(spatialFilter)[0] << ")\n";
    
    // Pasul 2: Calculăm transformata Fourier a filtrului spatial
    cout << "\nPasul 2: Calcularea TF a filtrului spatial\n";
    Mat filterFrequency = spatialFilterToFrequencyDomain(spatialFilter, image.rows, image.cols);
    cout << "TF a filtrului calculată cu dimensiuni: " << filterFrequency.rows << "x" << filterFrequency.cols << "\n";
    
    // Pasul 3: Calculăm DFT a imaginii
    cout << "\nPasul 3: Calcularea DFT a imaginii\n";
    Mat imageComplex = computeDFT(image);
    Mat imageShifted = shiftDFT(imageComplex);
    
    // Pasul 4: Înmulțim TF a imaginii cu TF a filtrului
    cout << "\nPasul 4: Înmulțirea TF a imaginii cu TF a filtrului\n";
    Mat filtered = multiplyComplexWithFilter(imageShifted, filterFrequency);
    
    // Pasul 5: Calculăm IDFT pentru a obține imaginea filtrată
    cout << "\nPasul 5: Calcularea IDFT (transformata inversă)\n";
    Mat shiftedBack = shiftDFT(filtered);
    Mat resultFrequency = computeIDFT(shiftedBack);
    resultFrequency = resultFrequency(Rect(0, 0, image.cols, image.rows));
    
    // Aplicăm filtrul și în domeniul spatial pentru comparație
    cout << "\nPasul 6: Aplicarea filtrului în domeniul spatial pentru comparație\n";
    Mat resultSpatial;
    filter2D(image, resultSpatial, -1, spatialFilter);
    
    // Afișăm rezultatele
    Mat imageDisplay;
    if (image.channels() == 3) {
        cvtColor(image, imageDisplay, COLOR_BGR2GRAY);
    } else {
        imageDisplay = image.clone();
    }
    
    // Normalizăm filtrul pentru afișare
    Mat filterDisplay;
    resize(spatialFilter, filterDisplay, Size(100, 100), 0, 0, INTER_NEAREST);
    normalize(filterDisplay, filterDisplay, 0, 255, NORM_MINMAX);
    filterDisplay.convertTo(filterDisplay, CV_8U);
    resize(filterDisplay, filterDisplay, imageDisplay.size());
    
    Mat display1, display2;
    hconcat(imageDisplay, filterDisplay, display1);
    hconcat(resultFrequency, resultSpatial, display2);
    
    Mat display;
    vconcat(display1, display2, display);
    
    // Salvăm rezultatele
    imwrite("output_lab9/punctul2_original.png", imageDisplay);
    imwrite("output_lab9/punctul2_filtru_spatial.png", filterDisplay);
    imwrite("output_lab9/punctul2_rezultat_frecventa.png", resultFrequency);
    imwrite("output_lab9/punctul2_rezultat_spatial.png", resultSpatial);
    imwrite("output_lab9/punctul2_combinat.png", display);
    
    imshow("Punctul 2 - Filtru Gaussian Spatial", display);
    cout << "\nAfișare:\n";
    cout << "Rand 1: Original | Filtru Gaussian Spatial (5x5)\n";
    cout << "Rand 2: Rezultat Frecvență | Rezultat Spatial\n";
    cout << "\n✓ Salvat: output_lab9/punctul2_original.png\n";
    cout << "✓ Salvat: output_lab9/punctul2_filtru_spatial.png\n";
    cout << "✓ Salvat: output_lab9/punctul2_rezultat_frecventa.png\n";
    cout << "✓ Salvat: output_lab9/punctul2_rezultat_spatial.png\n";
    cout << "✓ Salvat: output_lab9/punctul2_combinat.png\n";
    cout << "\nObservație: Rezultatele ar trebui să fie similare!\n";
    cout << "Apasati orice tasta pentru a continua...\n";
    waitKey(0);
    destroyWindow("Punctul 2 - Filtru Gaussian Spatial");
}

// Punctul 3: Filtrarea în domeniul frecvență pornind de la un filtru gaussian în domeniul frecvență
void filterFromFrequencyGaussian(const Mat& image, float D0) {
    cout << "\n=== Punctul 3: Filtrare în domeniu frecvență pornind de la filtru gaussian în domeniul frecvență ===\n";
    cout << "Parametru D0 (raza de tăiere): " << D0 << "\n";
    
    // Pasul 1: Creăm filtrul gaussian direct în domeniul frecvență
    cout << "\nPasul 1: Crearea filtrului gaussian în domeniul frecvenței\n";
    cout << "Formula: H(u,v) = e^(-D²/(2*D0²))\n";
    Mat filterFrequency = createGaussianLowPassFilter(image.rows, image.cols, D0);
    cout << "Filtru creat cu dimensiuni: " << filterFrequency.rows << "x" << filterFrequency.cols << "\n";
    
    // Pasul 2: Calculăm DFT a imaginii
    cout << "\nPasul 2: Calcularea DFT a imaginii\n";
    Mat imageComplex = computeDFT(image);
    Mat imageShifted = shiftDFT(imageComplex);
    
    // Pasul 3: Înmulțim TF a imaginii cu filtrul
    cout << "\nPasul 3: Înmulțirea TF a imaginii cu filtrul gaussian\n";
    Mat filtered = multiplyComplexWithFilter(imageShifted, filterFrequency);
    
    // Pasul 4: Calculăm IDFT pentru a obține imaginea filtrată
    cout << "\nPasul 4: Calcularea IDFT (transformata inversă)\n";
    Mat shiftedBack = shiftDFT(filtered);
    Mat resultFrequency = computeIDFT(shiftedBack);
    resultFrequency = resultFrequency(Rect(0, 0, image.cols, image.rows));
    
    // Afișăm rezultatele
    Mat imageDisplay;
    if (image.channels() == 3) {
        cvtColor(image, imageDisplay, COLOR_BGR2GRAY);
    } else {
        imageDisplay = image.clone();
    }
    
    // Normalizăm filtrul pentru afișare
    Mat filterDisplay;
    normalize(filterFrequency, filterDisplay, 0, 255, NORM_MINMAX);
    filterDisplay.convertTo(filterDisplay, CV_8U);
    resize(filterDisplay, filterDisplay, imageDisplay.size());
    
    // Calculăm spectrul rezultatului
    Mat resultComplex;
    Mat resultSpectrum = computeFourierSpectrum(resultFrequency, resultComplex);
    resize(resultSpectrum, resultSpectrum, imageDisplay.size());
    
    Mat display1, display2;
    hconcat(imageDisplay, filterDisplay, display1);
    hconcat(resultFrequency, resultSpectrum, display2);
    
    Mat display;
    vconcat(display1, display2, display);
    
    // Salvăm rezultatele
    imwrite("output_lab9/punctul3_original.png", imageDisplay);
    imwrite("output_lab9/punctul3_filtru_gaussian.png", filterDisplay);
    imwrite("output_lab9/punctul3_imagine_netezita.png", resultFrequency);
    imwrite("output_lab9/punctul3_spectru_rezultat.png", resultSpectrum);
    imwrite("output_lab9/punctul3_combinat.png", display);
    
    imshow("Punctul 3 - Filtru Gaussian Frecvență", display);
    cout << "\nAfișare:\n";
    cout << "Rand 1: Original | Filtru Gaussian Frecvență\n";
    cout << "Rand 2: Imagine Netezită | Spectru Rezultat\n";
    cout << "\n✓ Salvat: output_lab9/punctul3_original.png\n";
    cout << "✓ Salvat: output_lab9/punctul3_filtru_gaussian.png\n";
    cout << "✓ Salvat: output_lab9/punctul3_imagine_netezita.png\n";
    cout << "✓ Salvat: output_lab9/punctul3_spectru_rezultat.png\n";
    cout << "✓ Salvat: output_lab9/punctul3_combinat.png\n";
    cout << "\nObservație: Imaginea este netezită (blurred)!\n";
    cout << "Apasati orice tasta pentru a continua...\n";
    waitKey(0);
    destroyWindow("Punctul 3 - Filtru Gaussian Frecvență");
}

int main(int argc, char** argv) {
    // Încarcă imaginea
    string imagePath = "Imagini_Laborator/house.pgm";
    
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        cerr << "Eroare: Nu s-a putut incarca imaginea: " << imagePath << endl;
        cerr << "Folositi: " << argv[0] << " [cale_imagine]" << endl;
        return -1;
    }
    
    cout << "Imaginea incarcata: " << imagePath << endl;
    cout << "Dimensiuni: " << image.cols << "x" << image.rows << endl;
    
    int choice;
    float D0;
    
    while (true) {
        displayMenu();
        cin >> choice;
        
        if (cin.fail()) {
            cin.clear();
            cin.ignore(10000, '\n');
            cout << "Optiune invalida! Introduceti un numar.\n";
            continue;
        }
        
        switch (choice) {
            case 0:
                cout << "La revedere!\n";
                return 0;
                
            case 1:
                showFourierTransform(image);
                break;
                
            case 2:
                filterFromSpatialGaussian(image);
                break;
                
            case 3:
                cout << "Introduceti raza de taiere D0 (recomandat 20-50): ";
                cin >> D0;
                if (cin.fail() || D0 <= 0) {
                    cin.clear();
                    cin.ignore(10000, '\n');
                    cout << "Valoare invalida! Se folosește D0 = 30\n";
                    D0 = 30.0f;
                }
                filterFromFrequencyGaussian(image, D0);
                break;
                
            default:
                cout << "Optiune invalida! Alegeti intre 0-3.\n";
        }
    }
    
    return 0;
}
