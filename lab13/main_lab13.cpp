#include "color_spaces.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Variabile globale
Mat g_originalImage;
string g_outputDir = "output_lab13";

void printMenu() {
  cout << "\n====== LABORATOR 13 - Modele de Culoare ======" << endl;
  cout << "----------------------------------------------" << endl;
  cout << "1.  Aplicația 1: Egalizare histogramă (RGB vs HSI)" << endl;
  cout << "2.  Aplicația 2: Netezire filtru medie (RGB vs HSI)" << endl;
  cout << "3. Aplicația 3: Accentuare Laplacian (RGB)" << endl;
  cout << "----------------------------------------------" << endl;
  cout << "4. Rulează TOATE aplicațiile și salvează" << endl;
  cout << "0.  Ieșire" << endl;
  cout << "==============================================" << endl;
  cout << "Alegeți opțiunea: ";
}

void exercitiul1() {
  cout << "\n=== APLICAȚIA 1: EGALIZARE HISTOGRAMĂ (RGB vs HSI) ===" << endl;

  // Egalizare în RGB
  Mat equalizedRGB = equalizeHistogramRGB(g_originalImage);
  saveImage(equalizedRGB, g_outputDir + "/App1_equalized_RGB.png");
  cout << "  ✓ Egalizare RGB salvată" << endl;

  // Egalizare în HSI (doar pe I)
  Mat equalizedHSI = equalizeHistogramHSI(g_originalImage);
  saveImage(equalizedHSI, g_outputDir + "/App1_equalized_HSI.png");
  cout << "  ✓ Egalizare HSI salvată" << endl;

  // Comparație vizuală
  cout << "\n  OBSERVAȚII:" << endl;
  cout << "  - RGB: Egalizarea pe fiecare canal poate introduce artefacte de "
          "culoare"
       << endl;
  cout << "  - HSI: Egalizarea doar pe I păstrează nuanțele originale" << endl;

  imshow("Original", g_originalImage);
  imshow("Egalizat RGB (pe componente)", equalizedRGB);
  imshow("Egalizat HSI (pe I)", equalizedHSI);

  waitKey(0);
  destroyAllWindows();
}

void exercitiul2() {
  cout << "\n=== APLICAȚIA 2: NETEZIRE FILTRU MEDIE (RGB vs HSI) ===" << endl;

  vector<int> kernelSizes = {5, 10, 25};

  for (int size : kernelSizes) {
    cout << "\n  Kernel " << size << "x" << size << ":" << endl;

    // Netezire în RGB
    Mat smoothedRGB = smoothingRGB(g_originalImage, size);
    string filenameRGB =
        g_outputDir + "/App2_smooth_RGB_" + to_string(size) + ".png";
    saveImage(smoothedRGB, filenameRGB);

    // Netezire în HSI
    Mat smoothedHSI = smoothingHSI(g_originalImage, size);
    string filenameHSI =
        g_outputDir + "/App2_smooth_HSI_" + to_string(size) + ".png";
    saveImage(smoothedHSI, filenameHSI);

    // Afișare
    string windowRGB = "RGB " + to_string(size) + "x" + to_string(size);
    string windowHSI = "HSI " + to_string(size) + "x" + to_string(size);
    imshow(windowRGB, smoothedRGB);
    imshow(windowHSI, smoothedHSI);
  }

  cout << "\n  OBSERVAȚII:" << endl;
  cout << "  - RGB: Netezirea amestecă culorile, poate introduce culori false "
          "la margini"
       << endl;
  cout << "  - HSI: Netezirea pe I păstrează nuanțele, doar intensitatea se "
          "netezește"
       << endl;

  imshow("Original", g_originalImage);

  waitKey(0);
  destroyAllWindows();
}

void exercitiul3() {
  cout << "\n=== APLICAȚIA 3: ACCENTUARE LAPLACIAN (RGB) ===" << endl;

  Mat sharpened = sharpeningLaplacianRGB(g_originalImage);
  saveImage(sharpened, g_outputDir + "/App3_sharpened_Laplacian.png");

  // Calculăm și afișăm Laplacianul propriu-zis
  Mat gray;
  cvtColor(g_originalImage, gray, COLOR_BGR2GRAY);
  Mat laplacian;
  Laplacian(gray, laplacian, CV_16S, 3);
  Mat laplacianAbs;
  convertScaleAbs(laplacian, laplacianAbs);
  saveImage(laplacianAbs, g_outputDir + "/App3_Laplacian_edges.png");

  cout << "  ✓ Imagine accentuată salvată" << endl;
  cout << "  ✓ Marginile Laplacian salvate" << endl;

  imshow("Original", g_originalImage);
  imshow("Accentuată (Laplacian)", sharpened);
  imshow("Laplacian (margini)", laplacianAbs);

  waitKey(0);
  destroyAllWindows();
}

void exercitiul4() {
  cout << "\n========================================" << endl;
  cout << "  RULARE TOATE APLICAȚIILE" << endl;
  cout << "========================================\n" << endl;

  // Aplicația 1: Egalizare histogramă
  cout << "[1/3] Egalizare histogramă..." << endl;
  Mat equalizedRGB = equalizeHistogramRGB(g_originalImage);
  saveImage(equalizedRGB, g_outputDir + "/App1_equalized_RGB.png");
  Mat equalizedHSI = equalizeHistogramHSI(g_originalImage);
  saveImage(equalizedHSI, g_outputDir + "/App1_equalized_HSI.png");

  // Aplicația 2: Netezire
  cout << "[2/3] Netezire filtru medie..." << endl;
  vector<int> kernelSizes = {5, 10, 25};
  for (int size : kernelSizes) {
    Mat smoothedRGB = smoothingRGB(g_originalImage, size);
    saveImage(smoothedRGB,
              g_outputDir + "/App2_smooth_RGB_" + to_string(size) + ".png");
    Mat smoothedHSI = smoothingHSI(g_originalImage, size);
    saveImage(smoothedHSI,
              g_outputDir + "/App2_smooth_HSI_" + to_string(size) + ".png");
  }

  // Aplicația 3: Accentuare
  cout << "[3/3] Accentuare Laplacian..." << endl;
  Mat sharpened = sharpeningLaplacianRGB(g_originalImage);
  saveImage(sharpened, g_outputDir + "/App3_sharpened_Laplacian.png");

  // Salvăm și canalele spațiilor de culoare
  cout << "\n[Bonus] Salvare canale spații de culoare..." << endl;

  // RGB channels
  vector<Mat> rgbChannels;
  split(g_originalImage, rgbChannels);
  saveImage(rgbChannels[2], g_outputDir + "/RGB_R.png");
  saveImage(rgbChannels[1], g_outputDir + "/RGB_G.png");
  saveImage(rgbChannels[0], g_outputDir + "/RGB_B.png");

  // HSI components
  Mat hsiImage = rgb2hsi(g_originalImage);
  vector<Mat> hsiChannels;
  split(hsiImage, hsiChannels);
  Mat H_display, S_display, I_display;
  hsiChannels[0].convertTo(H_display, CV_8U, 255.0);
  hsiChannels[1].convertTo(S_display, CV_8U, 255.0);
  hsiChannels[2].convertTo(I_display, CV_8U, 255.0);
  saveImage(H_display, g_outputDir + "/HSI_H.png");
  saveImage(S_display, g_outputDir + "/HSI_S.png");
  saveImage(I_display, g_outputDir + "/HSI_I.png");

  cout << "\n========================================" << endl;
  cout << "  TOATE REZULTATELE AU FOST SALVATE!" << endl;
  cout << "  Verificați directorul: " << g_outputDir << "/" << endl;
  cout << "========================================\n" << endl;
}

int main(int argc, char **argv) {
  cout << "============================================" << endl;
  cout << "   LABORATOR 13 - Modele de Culoare        " << endl;
  cout << "============================================\n" << endl;

  // Încărcăm imaginea
  string imagePath;
  if (argc > 1) {
    imagePath = argv[1];
  } else {
    cout << "Utilizare: " << argv[0] << " <cale_imagine>" << endl;
    cout << "Exemplu: " << argv[0] << " Imagini_Laborator/lena.png" << endl;
    return -1;
  }

  // Citim imaginea
  g_originalImage = imread(imagePath);
  if (g_originalImage.empty()) {
    cerr << "Eroare: Nu se poate încărca imaginea: " << imagePath << endl;
    return -1;
  }

  cout << "Imagine încărcată: " << imagePath << endl;
  cout << "Dimensiuni: " << g_originalImage.cols << " x "
       << g_originalImage.rows << endl;
  cout << "Canale: " << g_originalImage.channels() << endl;

  // Creăm directorul de output
  fs::create_directories(g_outputDir);

  // Salvăm originalul
  saveImage(g_originalImage, g_outputDir + "/original.png");

  // Meniu interactiv
  int choice;
  do {
    printMenu();
    cin >> choice;

    switch (choice) {
    case 1:
      exercitiul1();
      break;
    case 2:
      exercitiul2();
      break;
    case 3:
      exercitiul3();
      break;
    case 4:
      exercitiul4();
      break;
    case 0:
      cout << "La revedere!" << endl;
      break;
    default:
      cout << "Opțiune invalidă!" << endl;
    }
  } while (choice != 0);

  return 0;
}
