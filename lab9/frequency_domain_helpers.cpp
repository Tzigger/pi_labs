#include "frequency_domain_helpers.h"
#include <iostream>

// ==================== FUNCȚII UTILITARE ====================

float computeDistance(int u, int v, int centerU, int centerV) {
    return std::sqrt((u - centerU) * (u - centerU) + (v - centerV) * (v - centerV));
}

cv::Mat getOptimalDFTSize(const cv::Mat& src) {
    int M = cv::getOptimalDFTSize(src.rows);
    int N = cv::getOptimalDFTSize(src.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    return padded;
}

// Convertește un filtru spatial într-un filtru în domeniul frecvenței
cv::Mat spatialFilterToFrequencyDomain(const cv::Mat& spatialFilter, int targetRows, int targetCols) {
    // Creăm o matrice mare cu dimensiunile dorite, centrată cu filtrul
    cv::Mat paddedFilter = cv::Mat::zeros(targetRows, targetCols, CV_32F);
    
    // Calculăm poziția de start pentru a centra filtrul
    int startRow = (targetRows - spatialFilter.rows) / 2;
    int startCol = (targetCols - spatialFilter.cols) / 2;
    
    // Copiem filtrul în centrul matricei
    cv::Mat roi = paddedFilter(cv::Rect(startCol, startRow, spatialFilter.cols, spatialFilter.rows));
    spatialFilter.copyTo(roi);
    
    // Convertim la float dacă nu este deja
    paddedFilter.convertTo(paddedFilter, CV_32F);
    
    // Creăm imaginea complexă pentru DFT
    cv::Mat planes[] = {paddedFilter, cv::Mat::zeros(paddedFilter.size(), CV_32F)};
    cv::Mat complexFilter;
    cv::merge(planes, 2, complexFilter);
    
    // Calculăm DFT
    cv::dft(complexFilter, complexFilter);
    
    // Shiftăm pentru a avea frecvențele nule în centru
    cv::Mat shifted = complexFilter.clone();
    int cx = shifted.cols / 2;
    int cy = shifted.rows / 2;
    
    cv::Mat q0(shifted, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(shifted, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(shifted, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(shifted, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    return shifted;
}

// ==================== TRANSFORMATA FOURIER ====================

cv::Mat computeDFT(const cv::Mat& src) {
    // Convertim imaginea la grayscale dacă este necesar
    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    // Optimizăm dimensiunea pentru DFT folosind funcția OpenCV
    int M = cv::getOptimalDFTSize(gray.rows);
    int N = cv::getOptimalDFTSize(gray.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 0, M - gray.rows, 0, N - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Convertim la float
    padded.convertTo(padded, CV_32F);
    
    // Creăm un array cu 2 canale (real, imaginar) folosind cv::merge
    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);
    
    // Aplicăm DFT folosind funcția OpenCV
    cv::dft(complexImage, complexImage);
    
    return complexImage;
}

cv::Mat shiftDFT(const cv::Mat& complexImage) {
    cv::Mat shifted = complexImage.clone();
    
    int cx = shifted.cols / 2;
    int cy = shifted.rows / 2;
    
    // Creăm ROI-uri pentru cele 4 cadrane
    cv::Mat q0(shifted, cv::Rect(0, 0, cx, cy));       // Top-Left
    cv::Mat q1(shifted, cv::Rect(cx, 0, cx, cy));      // Top-Right
    cv::Mat q2(shifted, cv::Rect(0, cy, cx, cy));      // Bottom-Left
    cv::Mat q3(shifted, cv::Rect(cx, cy, cx, cy));     // Bottom-Right
    
    // Interschimbăm cadranele
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    return shifted;
}

cv::Mat computeMagnitudeSpectrum(const cv::Mat& complexImage) {
    // Separăm canalele (real și imaginar) folosind cv::split
    cv::Mat planes[2];
    cv::split(complexImage, planes);
    
    // Calculăm magnitudinea folosind cv::magnitude: sqrt(Re^2 + Im^2)
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    
    // Adăugăm 1 pentru a evita log(0)
    magnitude += cv::Scalar::all(1);
    
    // Aplicăm logaritmul pentru o vizualizare mai bună folosind cv::log
    cv::log(magnitude, magnitude);
    
    return magnitude;
}

cv::Mat computeFourierSpectrum(const cv::Mat& src, cv::Mat& complexImage) {
    // Calculăm DFT folosind funcția noastră care apelează cv::dft
    complexImage = computeDFT(src);
    
    // Shiftăm pentru a avea frecvențele nule în centru
    cv::Mat shiftedComplex = shiftDFT(complexImage);
    
    // Calculăm spectrul de magnitudine folosind cv::magnitude și cv::log
    cv::Mat spectrum = computeMagnitudeSpectrum(shiftedComplex);
    
    // Normalizăm pentru afișare (0-255) folosind cv::normalize
    cv::Mat normalizedSpectrum;
    cv::normalize(spectrum, normalizedSpectrum, 0, 255, cv::NORM_MINMAX);
    normalizedSpectrum.convertTo(normalizedSpectrum, CV_8U);
    
    return normalizedSpectrum;
}

cv::Mat computeIDFT(const cv::Mat& complexImage) {
    cv::Mat inverseTransform;
    // Aplicăm IDFT folosind funcția OpenCV cu flagurile corespunzătoare
    cv::idft(complexImage, inverseTransform, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    // Extragem partea reală folosind cv::split
    cv::Mat planes[2];
    cv::split(inverseTransform, planes);
    
    // Normalizăm rezultatul folosind cv::normalize
    cv::Mat result;
    cv::normalize(planes[0], result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);
    
    return result;
}

// ==================== FILTRE TRECE-JOS ====================

cv::Mat createIdealLowPassFilter(int rows, int cols, float D0) {
    cv::Mat filter = cv::Mat::zeros(rows, cols, CV_32F);
    
    int centerU = cols / 2;
    int centerV = rows / 2;
    
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            float D = computeDistance(u, v, centerU, centerV);
            
            if (D <= D0) {
                filter.at<float>(v, u) = 1.0f;
            }
        }
    }
    
    return filter;
}

cv::Mat createButterworthLowPassFilter(int rows, int cols, float D0, int n) {
    cv::Mat filter = cv::Mat::zeros(rows, cols, CV_32F);
    
    int centerU = cols / 2;
    int centerV = rows / 2;
    
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            float D = computeDistance(u, v, centerU, centerV);
            
            // Formula Butterworth: H(u,v) = 1 / (1 + (D/D0)^(2n))
            float value = 1.0f / (1.0f + std::pow(D / D0, 2.0f * n));
            filter.at<float>(v, u) = value;
        }
    }
    
    return filter;
}

cv::Mat createGaussianLowPassFilter(int rows, int cols, float D0) {
    cv::Mat filter = cv::Mat::zeros(rows, cols, CV_32F);
    
    int centerU = cols / 2;
    int centerV = rows / 2;
    
    for (int v = 0; v < rows; v++) {
        for (int u = 0; u < cols; u++) {
            float D = computeDistance(u, v, centerU, centerV);
            
            // Formula Gaussian: H(u,v) = e^(-D^2 / (2*D0^2))
            float value = std::exp(-(D * D) / (2.0f * D0 * D0));
            filter.at<float>(v, u) = value;
        }
    }
    
    return filter;
}

// ==================== FILTRE TRECE-SUS ====================

cv::Mat createIdealHighPassFilter(int rows, int cols, float D0) {
    // H_hp(u,v) = 1 - H_lp(u,v)
    cv::Mat lowPassFilter = createIdealLowPassFilter(rows, cols, D0);
    return cv::Scalar::all(1.0f) - lowPassFilter;
}

cv::Mat createButterworthHighPassFilter(int rows, int cols, float D0, int n) {
    // H_hp(u,v) = 1 - H_lp(u,v)
    cv::Mat lowPassFilter = createButterworthLowPassFilter(rows, cols, D0, n);
    return cv::Scalar::all(1.0f) - lowPassFilter;
}

cv::Mat createGaussianHighPassFilter(int rows, int cols, float D0) {
    // H_hp(u,v) = 1 - H_lp(u,v)
    cv::Mat lowPassFilter = createGaussianLowPassFilter(rows, cols, D0);
    return cv::Scalar::all(1.0f) - lowPassFilter;
}

// ==================== APLICARE FILTRE ====================

cv::Mat multiplyComplexWithFilter(const cv::Mat& complexImage, const cv::Mat& filter) {
    // Separăm părțile real și imaginar folosind cv::split
    cv::Mat planes[2];
    cv::split(complexImage, planes);
    
    // Dacă filtrul are 2 canale (complex), multiplicăm complex
    if (filter.channels() == 2) {
        cv::Mat filterPlanes[2];
        cv::split(filter, filterPlanes);
        
        // Multiplicare complexă: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        // Folosim cv::multiply pentru multiplicarea element cu element
        cv::Mat ac, bd, ad, bc;
        cv::multiply(planes[0], filterPlanes[0], ac);
        cv::multiply(planes[1], filterPlanes[1], bd);
        cv::multiply(planes[0], filterPlanes[1], ad);
        cv::multiply(planes[1], filterPlanes[0], bc);
        
        cv::Mat real = ac - bd;
        cv::Mat imag = ad + bc;
        
        cv::Mat result;
        cv::Mat resultPlanes[] = {real, imag};
        cv::merge(resultPlanes, 2, result);
        
        return result;
    } else {
        // Multiplicare cu filtru real folosind cv::multiply
        cv::Mat filteredReal, filteredImag;
        cv::multiply(planes[0], filter, filteredReal);
        cv::multiply(planes[1], filter, filteredImag);
        
        cv::Mat result;
        cv::Mat resultPlanes[] = {filteredReal, filteredImag};
        cv::merge(resultPlanes, 2, result);
        
        return result;
    }
}

cv::Mat applyFrequencyFilter(const cv::Mat& src, const cv::Mat& filter) {
    // Pasul 1: Calculăm DFT
    cv::Mat complexImage = computeDFT(src);
    
    // Shiftăm pentru a avea frecvențele nule în centru
    cv::Mat shiftedComplex = shiftDFT(complexImage);
    
    // Pasul 2: Multiplicăm cu filtrul
    cv::Mat filtered = multiplyComplexWithFilter(shiftedComplex, filter);
    
    // Shiftăm înapoi
    cv::Mat shiftedBack = shiftDFT(filtered);
    
    // Pasul 3: Aplicăm IDFT
    cv::Mat result = computeIDFT(shiftedBack);
    
    // Redimensionăm la dimensiunea originală
    result = result(cv::Rect(0, 0, src.cols, src.rows));
    
    return result;
}
