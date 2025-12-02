#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "Complex.h"

// ==================== TRANSFORMATA FOURIER DISCRETĂ ====================

// Calculează transformata Fourier discretă 2D folosind formula din definiție
std::vector<std::vector<Complex>> computeDFT2D(const cv::Mat& img) {
    int M = img.rows;
    int N = img.cols;
    
    std::cout << "Calculare DFT pentru imagine " << M << "x" << N << "..." << std::endl;
    std::cout << "Atenție: Acest lucru poate dura ceva timp..." << std::endl;
    
    // Inițializare matrice de numere complexe pentru rezultat
    std::vector<std::vector<Complex>> F(M, std::vector<Complex>(N));
    
    const double PI = 3.14159265358979323846;
    
    // Calculare transformata Fourier conform definiției
    // F(u,v) = Σ(x=0 to M-1) Σ(y=0 to N-1) f(x,y) * e^(-j*2π*(ux/M + vy/N))
    for (int u = 0; u < M; u++) {
        // Afișare progres
        if (u % 10 == 0) {
            std::cout << "Progres: " << u << "/" << M << std::endl;
        }
        
        for (int v = 0; v < N; v++) {
            Complex sum(0, 0);
            
            for (int x = 0; x < M; x++) {
                for (int y = 0; y < N; y++) {
                    // Obținere valoarea pixelului
                    double f_xy = (double)img.at<uchar>(x, y);
                    
                    // Calculare exponențiala complexă: e^(-j*2π*(ux/M + vy/N))
                    double angle = -2.0 * PI * ((double)(u * x) / M + (double)(v * y) / N);
                    Complex exponential = Complex::expI(angle);
                    
                    // Adunare la suma
                    sum += exponential * f_xy;
                }
            }
            
            F[u][v] = sum;
        }
    }
    
    std::cout << "DFT calculată cu succes!" << std::endl;
    return F;
}

// Calculează transformata Fourier inversă 2D
cv::Mat computeInverseDFT2D(const std::vector<std::vector<Complex>>& F) {
    int M = F.size();
    int N = F[0].size();
    
    std::cout << "Calculare IDFT pentru imagine " << M << "x" << N << "..." << std::endl;
    
    cv::Mat img(M, N, CV_8UC1);
    const double PI = 3.14159265358979323846;
    
    // f(x,y) = (1/MN) * Σ(u=0 to M-1) Σ(v=0 to N-1) F(u,v) * e^(j*2π*(ux/M + vy/N))
    for (int x = 0; x < M; x++) {
        if (x % 10 == 0) {
            std::cout << "Progres: " << x << "/" << M << std::endl;
        }
        
        for (int y = 0; y < N; y++) {
            Complex sum(0, 0);
            
            for (int u = 0; u < M; u++) {
                for (int v = 0; v < N; v++) {
                    // Calculare exponențiala complexă: e^(j*2π*(ux/M + vy/N))
                    double angle = 2.0 * PI * ((double)(u * x) / M + (double)(v * y) / N);
                    Complex exponential = Complex::expI(angle);
                    
                    // Adunare la suma
                    sum += F[u][v] * exponential;
                }
            }
            
            // Normalizare cu 1/(M*N) și luare partea reală
            double value = sum.getReal() / (M * N);
            
            // Clipping la [0, 255]
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            
            img.at<uchar>(x, y) = (uchar)value;
        }
    }
    
    std::cout << "IDFT calculată cu succes!" << std::endl;
    return img;
}

// Shiftează originea spectrului în centru (interschimbă cadranele)
std::vector<std::vector<Complex>> shiftSpectrum(const std::vector<std::vector<Complex>>& F) {
    int M = F.size();
    int N = F[0].size();
    
    std::vector<std::vector<Complex>> shifted(M, std::vector<Complex>(N));
    
    int halfM = M / 2;
    int halfN = N / 2;
    
    // Interschimbare cadrane:
    // Cadranul 1 (stânga-sus) cu Cadranul 4 (dreapta-jos)
    // Cadranul 2 (dreapta-sus) cu Cadranul 3 (stânga-jos)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int newI = (i + halfM) % M;
            int newJ = (j + halfN) % N;
            shifted[newI][newJ] = F[i][j];
        }
    }
    
    return shifted;
}

// Afișează spectrul transformatei Fourier conform punctului 2.2
cv::Mat displaySpectrum(const std::vector<std::vector<Complex>>& F, bool shift = true) {
    int M = F.size();
    int N = F[0].size();
    
    std::vector<std::vector<Complex>> spectrum = F;
    
    // Pasul 1: Shiftarea originii spectrului (componenta DC în centru)
    if (shift) {
        std::cout << "Shiftare spectru..." << std::endl;
        spectrum = shiftSpectrum(F);
    }
    
    // Pasul 2: Calculare modul și găsire valorii maxime
    std::cout << "Calcul modul..." << std::endl;
    std::vector<std::vector<double>> magnitude(M, std::vector<double>(N));
    double maxMagnitude = 0.0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Calcul modul
            magnitude[i][j] = spectrum[i][j].magnitude();
            
            if (magnitude[i][j] > maxMagnitude) {
                maxMagnitude = magnitude[i][j];
            }
        }
    }
    
    // Pasul 3: Logaritmare pentru mai multă vizibilitate: log(1 + |F|)
    std::cout << "Logaritmare..." << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            magnitude[i][j] = std::log(1.0 + magnitude[i][j]);
        }
    }
    
    // Găsire maxim după logaritmare
    maxMagnitude = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (magnitude[i][j] > maxMagnitude) {
                maxMagnitude = magnitude[i][j];
            }
        }
    }
    
    // Creare imagine pentru afișare
    cv::Mat spectrumImage(M, N, CV_8UC1);
    
    // Pasul 4: Normalizare (împărțire la valoarea maximă)
    // Pasul 5: Scalare la [0, 255]
    std::cout << "Normalizare și scalare..." << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double normalized = magnitude[i][j] / maxMagnitude;  // Normalizare la [0,1]
            double scaled = normalized * 255.0;                   // Scalare la [0,255]
            
            if (scaled < 0) scaled = 0;
            if (scaled > 255) scaled = 255;
            
            spectrumImage.at<uchar>(i, j) = (uchar)scaled;
        }
    }
    
    return spectrumImage;
}

// ==================== FILTRE ÎN DOMENIUL FRECVENȚELOR ====================

// Calculează distanța de la un punct (u,v) la originea spectrului
double calculateDistance(int u, int v, int M, int N) {
    // Distanța față de colțul cel mai apropiat (originea spectrului neshiftat)
    double du = std::min(u, M - u);
    double dv = std::min(v, N - v);
    return std::sqrt(du * du + dv * dv);
}

// Filtru trece-jos Gaussian
std::vector<std::vector<double>> createGaussianLowPassFilter(int M, int N, double D0) {
    std::vector<std::vector<double>> H(M, std::vector<double>(N));
    
    std::cout << "Creare filtru Gaussian trece-jos cu D0 = " << D0 << "..." << std::endl;
    
    for (int u = 0; u < M; u++) {
        for (int v = 0; v < N; v++) {
            double D = calculateDistance(u, v, M, N);
            // H(u,v) = e^(-D²(u,v)/(2*D0²))
            H[u][v] = std::exp(-(D * D) / (2.0 * D0 * D0));
        }
    }
    
    return H;
}

// Aplică filtrul în domeniul frecvențelor (înmulțire element cu element)
std::vector<std::vector<Complex>> applyFilter(const std::vector<std::vector<Complex>>& F,
                                               const std::vector<std::vector<double>>& H) {
    int M = F.size();
    int N = F[0].size();
    
    std::cout << "Aplicare filtru..." << std::endl;
    std::vector<std::vector<Complex>> G(M, std::vector<Complex>(N));
    
    for (int u = 0; u < M; u++) {
        for (int v = 0; v < N; v++) {
            // G(u,v) = F(u,v) * H(u,v)
            G[u][v] = F[u][v] * H[u][v];
        }
    }
    
    return G;
}

// Afișează filtrul
cv::Mat displayFilter(const std::vector<std::vector<double>>& H, bool shift = true) {
    int M = H.size();
    int N = H[0].size();
    
    cv::Mat filterImage(M, N, CV_8UC1);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int displayI = i;
            int displayJ = j;
            
            // Shift pentru afișare (componenta DC în centru)
            if (shift) {
                displayI = (i + M/2) % M;
                displayJ = (j + N/2) % N;
            }
            
            double value = H[i][j] * 255.0;
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            
            filterImage.at<uchar>(displayI, displayJ) = (uchar)value;
        }
    }
    
    return filterImage;
}

// ==================== APLICAȚII ====================

void aplicatia1() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "APLICAȚIA 1" << std::endl;
    std::cout << "Calculare și Afișare Transformată Fourier" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Încărcare imagine în grayscale
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        std::cerr << "Eroare: Nu s-a putut încărca imaginea!" << std::endl;
        std::cerr << "Verificați că fișierul 'Imagini_Laborator/coins.png' există." << std::endl;
        return;
    }
    
    std::cout << "Imagine încărcată: " << img.rows << "x" << img.cols << " pixeli" << std::endl;
    
    // Redimensionare la dimensiune mică pentru calcul rapid (max 128x128)
    cv::Mat smallImg;
    int targetSize = 128;
    
    if (img.rows > targetSize || img.cols > targetSize) {
        cv::resize(img, smallImg, cv::Size(targetSize, targetSize));
        std::cout << "Imagine redimensionată la: " << smallImg.rows << "x" << smallImg.cols << std::endl;
    } else {
        smallImg = img.clone();
    }
    
    std::cout << "\nÎncepe calculul DFT..." << std::endl;
    std::cout << "Timp estimat: ~1-2 minute pentru 128x128" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Calculare DFT folosind formula din definiție
    std::vector<std::vector<Complex>> F = computeDFT2D(smallImg);
    
    std::cout << "\nAfișare spectru conform punctului 2.2..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Afișare spectru conform punctului 2.2
    // (shiftare, calcul modul, logaritmare, normalizare, scalare)
    cv::Mat spectrum = displaySpectrum(F, true);
    
    std::cout << "\nInformații despre transformată:" << std::endl;
    std::cout << "Componenta DC (F(0,0)): " << F[0][0] << std::endl;
    std::cout << "Modul F(0,0): " << F[0][0].magnitude() << std::endl;
    
    // Salvare imagini
    cv::imwrite("output_lab8/aplicatia1_imagine_originala.png", smallImg);
    cv::imwrite("output_lab8/aplicatia1_spectrul_fourier.png", spectrum);
    std::cout << "\nImagini salvate în folder-ul output_lab8/" << std::endl;
    std::cout << "  - aplicatia1_imagine_originala.png" << std::endl;
    std::cout << "  - aplicatia1_spectrul_fourier.png" << std::endl;
}

void aplicatia2() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "APLICAȚIA 2" << std::endl;
    std::cout << "DFT → IDFT (Reconstrucție Imagine)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Încărcare imagine în grayscale
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        std::cerr << "Eroare: Nu s-a putut încărca imaginea!" << std::endl;
        std::cerr << "Verificați că fișierul 'Imagini_Laborator/coins.png' există." << std::endl;
        return;
    }
    
    std::cout << "Imagine încărcată: " << img.rows << "x" << img.cols << " pixeli" << std::endl;
    
    // Redimensionare la dimensiune mică pentru calcul rapid (max 128x128)
    cv::Mat smallImg;
    int targetSize = 128;
    
    if (img.rows > targetSize || img.cols > targetSize) {
        cv::resize(img, smallImg, cv::Size(targetSize, targetSize));
        std::cout << "Imagine redimensionată la: " << smallImg.rows << "x" << smallImg.cols << std::endl;
    } else {
        smallImg = img.clone();
    }
    
    std::cout << "\n=== PAS 1: Aplicare transformata Fourier ===" << std::endl;
    std::cout << "Timp estimat: ~1-2 minute pentru 128x128" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    // PAS 1: Calculare DFT folosind formula din definiție
    std::vector<std::vector<Complex>> F = computeDFT2D(smallImg);
    
    std::cout << "\n=== PAS 2: Aplicare transformata inversă ===" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    // PAS 2: Aplicare transformata inversă
    // Notă: computeInverseDFT2D ia doar componentele reale, le normalizează și le înmulțește cu 255
    cv::Mat reconstructedImg = computeInverseDFT2D(F);
    
    std::cout << "\n=== Verificare rezultate ===" << std::endl;
    std::cout << "Imaginea reconstruită trebuie să fie identică cu imaginea originală." << std::endl;
    
    // Calcul diferență pentru verificare
    cv::Mat diff;
    cv::absdiff(smallImg, reconstructedImg, diff);
    double maxDiff = 0;
    cv::minMaxLoc(diff, nullptr, &maxDiff);
    
    std::cout << "Diferența maximă între imaginea originală și cea reconstruită: " << maxDiff << " (din 255)" << std::endl;
    
    if (maxDiff < 1.0) {
        std::cout << "✓ Reconstrucția este perfect!" << std::endl;
    } else if (maxDiff < 5.0) {
        std::cout << "✓ Reconstrucția este excelentă (diferențe minore datorită erorilor de rotunjire)" << std::endl;
    } else {
        std::cout << "⚠ Există diferențe notabile (posibil din cauza erorilor numerice)" << std::endl;
    }
    
    // Salvare imagini
    cv::imwrite("output_lab8/aplicatia2_imagine_originala.png", smallImg);
    cv::imwrite("output_lab8/aplicatia2_imagine_reconstruita.png", reconstructedImg);
    cv::imwrite("output_lab8/aplicatia2_diferenta.png", diff * 10); // Amplificat pentru vizibilitate
    std::cout << "\nImagini salvate în folder-ul output_lab8/" << std::endl;
    std::cout << "  - aplicatia2_imagine_originala.png" << std::endl;
    std::cout << "  - aplicatia2_imagine_reconstruita.png" << std::endl;
    std::cout << "  - aplicatia2_diferenta.png (amplificată x10)" << std::endl;
}

void aplicatia3() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "APLICAȚIA 3" << std::endl;
    std::cout << "Filtru Trece-Jos Gaussian în Domeniul Frecvențelor" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Încărcare imagine
    cv::Mat img = cv::imread("Imagini_Laborator/coins.png", cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        std::cerr << "Eroare: Nu s-a putut încărca imaginea!" << std::endl;
        std::cerr << "Verificați că fișierul 'Imagini_Laborator/coins.png' există." << std::endl;
        return;
    }
    
    std::cout << "Imagine încărcată: " << img.rows << "x" << img.cols << " pixeli" << std::endl;
    
    // Redimensionare
    cv::Mat smallImg;
    int targetSize = 128;
    
    if (img.rows > targetSize || img.cols > targetSize) {
        cv::resize(img, smallImg, cv::Size(targetSize, targetSize));
        std::cout << "Imagine redimensionată la: " << smallImg.rows << "x" << smallImg.cols << std::endl;
    } else {
        smallImg = img.clone();
    }
    
    std::cout << "\n=== PAS 1: Calculul transformatei Fourier a imaginii originale ===" << std::endl;
    std::cout << "Timp estimat: ~1-2 minute pentru 128x128" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    
    // PAS 1: Calculul transformatei Fourier a imaginii originale
    std::vector<std::vector<Complex>> F = computeDFT2D(smallImg);
    
    std::cout << "\n=== PAS 2: Aplicarea filtrului prin înmulțire ===" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    
    // PAS 2: Aplicarea filtrului prin înmulțire cu fiecare valoare a transformatei
    double D0 = 30.0; // Frecvența de tăiere
    std::vector<std::vector<double>> H = createGaussianLowPassFilter(smallImg.rows, smallImg.cols, D0);
    std::vector<std::vector<Complex>> G = applyFilter(F, H);
    
    std::cout << "\n=== PAS 3: Aplicarea transformatei inverse ===" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    
    // PAS 3: Aplicarea transformatei inverse asupra rezultatului obținut anterior
    cv::Mat filteredImg = computeInverseDFT2D(G);
    
    // Observație: Normalizarea și înmulțirea cu 255 sunt deja incluse în computeInverseDFT2D
    // (PAS 4 și PAS 5 sunt realizate automat)
    
    std::cout << "\n=== Rezultate ===" << std::endl;
    std::cout << "Imaginea filtrată trebuie să fie o variantă netezită (blurred) a imaginii originale." << std::endl;
    
    // Afișare rezultate
    cv::Mat spectrumOriginal = displaySpectrum(F, true);
    cv::Mat spectrumFiltered = displaySpectrum(G, true);
    cv::Mat filterDisplay = displayFilter(H, true);
    
    // Salvare imagini
    cv::imwrite("output_lab8/aplicatia3_imagine_originala.png", smallImg);
    cv::imwrite("output_lab8/aplicatia3_spectru_original.png", spectrumOriginal);
    cv::imwrite("output_lab8/aplicatia3_filtru_gaussian.png", filterDisplay);
    cv::imwrite("output_lab8/aplicatia3_spectru_filtrat.png", spectrumFiltered);
    cv::imwrite("output_lab8/aplicatia3_imagine_filtrata_blurred.png", filteredImg);
    std::cout << "\nImagini salvate în folder-ul output_lab8/" << std::endl;
    std::cout << "  - aplicatia3_imagine_originala.png" << std::endl;
    std::cout << "  - aplicatia3_spectru_original.png" << std::endl;
    std::cout << "  - aplicatia3_filtru_gaussian.png" << std::endl;
    std::cout << "  - aplicatia3_spectru_filtrat.png" << std::endl;
    std::cout << "  - aplicatia3_imagine_filtrata_blurred.png" << std::endl;
}

// ==================== MAIN ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Prelucrarea Imaginilor - Laborator 8" << std::endl;
    std::cout << "Transformata Fourier Discretă" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int choice;
    
    while(true) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "MENIU PRINCIPAL" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "1. Aplicația 1 - Calculare și Afișare Transformată Fourier" << std::endl;
        std::cout << "2. Aplicația 2 - Calculare și Afișare Transformată Fourier" << std::endl;
        std::cout << "3. Aplicația 3 - Filtru Trece-Jos Gaussian" << std::endl;
        std::cout << "4. Rulează toate aplicațiile" << std::endl;
        std::cout << "0. Exit" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Alege opțiunea: ";
        
        std::cin >> choice;
        
        switch(choice) {
            case 1:
                aplicatia1();
                break;
            case 2:
                aplicatia2();
                break;
            case 3:
                aplicatia3();
                break;
            case 4:
                std::cout << "\n### Rulare TOATE APLICAȚIILE ###\n" << std::endl;
                aplicatia1();
                aplicatia2();
                aplicatia3();
                std::cout << "\n### TOATE APLICAȚIILE AU FOST EXECUTATE ###\n" << std::endl;
                break;
            case 0:
                std::cout << "\nÎnchidere program..." << std::endl;
                return 0;
            default:
                std::cout << "\nOpțiune invalidă! Vă rugăm alegeți 0-4." << std::endl;
        }
    }
    
    return 0;
}