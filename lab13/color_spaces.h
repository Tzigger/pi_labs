#ifndef COLOR_SPACES_H
#define COLOR_SPACES_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

// ============================================================================
// CONVERSII ÎNTRE SPAȚII DE CULOARE
// ============================================================================

/**
 * Conversie RGB -> CMY
 * Formula: C = 1 - R, M = 1 - G, Y = 1 - B (valori normalizate [0,1])
 */
Mat rgb2cmy(const Mat& rgbImage);

/**
 * Conversie CMY -> RGB
 * Formula: R = 1 - C, G = 1 - M, B = 1 - Y
 */
Mat cmy2rgb(const Mat& cmyImage);

/**
 * Conversie RGB -> CMYK
 * K = min(C, M, Y), apoi C = C - K, M = M - K, Y = Y - K
 */
void rgb2cmyk(const Mat& rgbImage, Mat& cmykImage);

/**
 * Conversie CMYK -> RGB
 */
Mat cmyk2rgb(const Mat& cmykImage);

/**
 * Conversie RGB -> YIQ (NTSC)
 * Matricea de transformare conform formulei (4)
 */
Mat rgb2yiq(const Mat& rgbImage);

/**
 * Conversie YIQ -> RGB
 */
Mat yiq2rgb(const Mat& yiqImage);

/**
 * Conversie RGB -> HSI
 * Formulele (5)-(8) din laborator
 */
Mat rgb2hsi(const Mat& rgbImage);

/**
 * Conversie HSI -> RGB
 * Formulele (9)-(19) din laborator, pe cele 3 sectoare
 */
Mat hsi2rgb(const Mat& hsiImage);

/**
 * Conversie RGB -> XYZ (CIE 1931)
 * Folosește matricea standard de transformare
 */
Mat rgb2xyz(const Mat& rgbImage);

/**
 * Conversie XYZ -> RGB
 */
Mat xyz2rgb(const Mat& xyzImage);

/**
 * Conversie RGB -> CIELAB (L*a*b*)
 * Formulele (20)-(23) din laborator
 */
Mat rgb2lab(const Mat& rgbImage);

/**
 * Conversie LAB -> RGB
 */
Mat lab2rgb(const Mat& labImage);

// ============================================================================
// APLICAȚII DE PROCESARE IMAGINI COLOR
// ============================================================================

/**
 * Aplicația 1a: Egalizare histogramă în spațiul RGB (pe componente separate)
 */
Mat equalizeHistogramRGB(const Mat& rgbImage);

/**
 * Aplicația 1b: Egalizare histogramă în spațiul HSI (doar pe componenta I)
 */
Mat equalizeHistogramHSI(const Mat& rgbImage);

/**
 * Aplicația 2a: Netezire în spațiul RGB (filtru medie pe componente)
 */
Mat smoothingRGB(const Mat& rgbImage, int kernelSize);

/**
 * Aplicația 2b: Netezire în spațiul HSI (filtru medie pe I)
 */
Mat smoothingHSI(const Mat& rgbImage, int kernelSize);

/**
 * Aplicația 3: Accentuare cu filtru Laplacian în spațiul RGB
 */
Mat sharpeningLaplacianRGB(const Mat& rgbImage);

// ============================================================================
// FUNCȚII AUXILIARE
// ============================================================================

/**
 * Funcția h(q) pentru conversie LAB - formula (23)
 */
double labFunction(double q);

/**
 * Funcția inversă pentru LAB
 */
double labFunctionInverse(double q);

/**
 * Vizualizare canale ale unui spațiu de culoare
 */
void visualizeColorChannels(const Mat& image, const string& spaceName, 
                            const vector<string>& channelNames, const string& outputDir);

/**
 * Salvare imagine cu mesaj
 */
void saveImage(const Mat& image, const string& filename);

#endif // COLOR_SPACES_H
