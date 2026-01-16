#include "color_spaces.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Constante pentru conversie LAB
const double X0 = 95.047;   // Iluminant D65
const double Y0 = 100.000;
const double Z0 = 108.883;

// ============================================================================
// CONVERSII CMY / CMYK
// ============================================================================

Mat rgb2cmy(const Mat& rgbImage) {
    Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_64FC3, 1.0/255.0);
    
    Mat cmyImage(rgbFloat.size(), CV_64FC3);
    
    for (int i = 0; i < rgbFloat.rows; i++) {
        for (int j = 0; j < rgbFloat.cols; j++) {
            Vec3d rgb = rgbFloat.at<Vec3d>(i, j);
            // OpenCV: BGR format
            double B = rgb[0], G = rgb[1], R = rgb[2];
            
            double C = 1.0 - R;
            double M = 1.0 - G;
            double Y = 1.0 - B;
            
            cmyImage.at<Vec3d>(i, j) = Vec3d(Y, M, C);  // YMC order for display
        }
    }
    
    return cmyImage;
}

Mat cmy2rgb(const Mat& cmyImage) {
    Mat rgbImage(cmyImage.size(), CV_64FC3);
    
    for (int i = 0; i < cmyImage.rows; i++) {
        for (int j = 0; j < cmyImage.cols; j++) {
            Vec3d cmy = cmyImage.at<Vec3d>(i, j);
            double Y = cmy[0], M = cmy[1], C = cmy[2];
            
            double R = 1.0 - C;
            double G = 1.0 - M;
            double B = 1.0 - Y;
            
            rgbImage.at<Vec3d>(i, j) = Vec3d(B, G, R);
        }
    }
    
    Mat result;
    rgbImage.convertTo(result, CV_8UC3, 255.0);
    return result;
}

void rgb2cmyk(const Mat& rgbImage, Mat& cmykImage) {
    Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_64FC3, 1.0/255.0);
    
    cmykImage = Mat(rgbFloat.size(), CV_64FC4);
    
    for (int i = 0; i < rgbFloat.rows; i++) {
        for (int j = 0; j < rgbFloat.cols; j++) {
            Vec3d rgb = rgbFloat.at<Vec3d>(i, j);
            double B = rgb[0], G = rgb[1], R = rgb[2];
            
            double C = 1.0 - R;
            double M = 1.0 - G;
            double Y = 1.0 - B;
            double K = min({C, M, Y});
            
            // Avoid division by zero
            if (K < 1.0) {
                C = C - K;
                M = M - K;
                Y = Y - K;
            } else {
                C = M = Y = 0;
            }
            
            cmykImage.at<Vec4d>(i, j) = Vec4d(C, M, Y, K);
        }
    }
}

Mat cmyk2rgb(const Mat& cmykImage) {
    Mat rgbImage(cmykImage.size(), CV_64FC3);
    
    for (int i = 0; i < cmykImage.rows; i++) {
        for (int j = 0; j < cmykImage.cols; j++) {
            Vec4d cmyk = cmykImage.at<Vec4d>(i, j);
            double C = cmyk[0], M = cmyk[1], Y = cmyk[2], K = cmyk[3];
            
            C = C + K;
            M = M + K;
            Y = Y + K;
            
            double R = 1.0 - C;
            double G = 1.0 - M;
            double B = 1.0 - Y;
            
            // Clamp values
            R = max(0.0, min(1.0, R));
            G = max(0.0, min(1.0, G));
            B = max(0.0, min(1.0, B));
            
            rgbImage.at<Vec3d>(i, j) = Vec3d(B, G, R);
        }
    }
    
    Mat result;
    rgbImage.convertTo(result, CV_8UC3, 255.0);
    return result;
}

// ============================================================================
// CONVERSIE YIQ (NTSC)
// ============================================================================

Mat rgb2yiq(const Mat& rgbImage) {
    Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_64FC3, 1.0/255.0);
    
    Mat yiqImage(rgbFloat.size(), CV_64FC3);
    
    // Matricea de transformare RGB -> YIQ (formula 4)
    // Y = 0.299*R + 0.587*G + 0.114*B
    // I = 0.596*R - 0.274*G - 0.322*B
    // Q = 0.211*R - 0.523*G + 0.312*B
    
    for (int i = 0; i < rgbFloat.rows; i++) {
        for (int j = 0; j < rgbFloat.cols; j++) {
            Vec3d rgb = rgbFloat.at<Vec3d>(i, j);
            double B = rgb[0], G = rgb[1], R = rgb[2];
            
            double Y = 0.299 * R + 0.587 * G + 0.114 * B;
            double I = 0.596 * R - 0.274 * G - 0.322 * B;
            double Q = 0.211 * R - 0.523 * G + 0.312 * B;
            
            yiqImage.at<Vec3d>(i, j) = Vec3d(Y, I, Q);
        }
    }
    
    return yiqImage;
}

Mat yiq2rgb(const Mat& yiqImage) {
    Mat rgbImage(yiqImage.size(), CV_64FC3);
    
    // Matricea inversă YIQ -> RGB
    for (int i = 0; i < yiqImage.rows; i++) {
        for (int j = 0; j < yiqImage.cols; j++) {
            Vec3d yiq = yiqImage.at<Vec3d>(i, j);
            double Y = yiq[0], I = yiq[1], Q = yiq[2];
            
            double R = Y + 0.956 * I + 0.621 * Q;
            double G = Y - 0.272 * I - 0.647 * Q;
            double B = Y - 1.106 * I + 1.703 * Q;
            
            // Clamp values
            R = max(0.0, min(1.0, R));
            G = max(0.0, min(1.0, G));
            B = max(0.0, min(1.0, B));
            
            rgbImage.at<Vec3d>(i, j) = Vec3d(B, G, R);
        }
    }
    
    Mat result;
    rgbImage.convertTo(result, CV_8UC3, 255.0);
    return result;
}

// ============================================================================
// CONVERSIE HSI
// ============================================================================

Mat rgb2hsi(const Mat& rgbImage) {
    Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_64FC3, 1.0/255.0);
    
    Mat hsiImage(rgbFloat.size(), CV_64FC3);
    
    for (int i = 0; i < rgbFloat.rows; i++) {
        for (int j = 0; j < rgbFloat.cols; j++) {
            Vec3d rgb = rgbFloat.at<Vec3d>(i, j);
            double B = rgb[0], G = rgb[1], R = rgb[2];
            
            // Intensity - formula (8)
            double I = (R + G + B) / 3.0;
            
            // Saturation - formula (7)
            double minRGB = min({R, G, B});
            double S = 0;
            if (I > 0) {
                S = 1.0 - (3.0 * minRGB / (R + G + B));
            }
            
            // Hue - formulas (5), (6)
            double H = 0;
            double num = 0.5 * ((R - G) + (R - B));
            double den = sqrt((R - G) * (R - G) + (R - B) * (G - B));
            
            if (den > 1e-10) {
                double theta = acos(max(-1.0, min(1.0, num / den)));
                if (B <= G) {
                    H = theta;
                } else {
                    H = 2.0 * CV_PI - theta;
                }
            }
            
            // Normalizare H în [0, 1] (H era în [0, 2*PI])
            H = H / (2.0 * CV_PI);
            
            hsiImage.at<Vec3d>(i, j) = Vec3d(H, S, I);
        }
    }
    
    return hsiImage;
}

Mat hsi2rgb(const Mat& hsiImage) {
    Mat rgbImage(hsiImage.size(), CV_64FC3);
    
    for (int i = 0; i < hsiImage.rows; i++) {
        for (int j = 0; j < hsiImage.cols; j++) {
            Vec3d hsi = hsiImage.at<Vec3d>(i, j);
            double H = hsi[0] * 360.0;  // Convert back to degrees
            double S = hsi[1];
            double I = hsi[2];
            
            double R, G, B;
            
            // Sector RG: 0 <= H < 120
            if (H >= 0 && H < 120) {
                B = I * (1.0 - S);
                R = I * (1.0 + (S * cos(H * CV_PI / 180.0)) / cos((60.0 - H) * CV_PI / 180.0));
                G = 3.0 * I - (R + B);
            }
            // Sector GB: 120 <= H < 240
            else if (H >= 120 && H < 240) {
                H = H - 120.0;
                R = I * (1.0 - S);
                G = I * (1.0 + (S * cos(H * CV_PI / 180.0)) / cos((60.0 - H) * CV_PI / 180.0));
                B = 3.0 * I - (R + G);
            }
            // Sector BR: 240 <= H < 360
            else {
                H = H - 240.0;
                G = I * (1.0 - S);
                B = I * (1.0 + (S * cos(H * CV_PI / 180.0)) / cos((60.0 - H) * CV_PI / 180.0));
                R = 3.0 * I - (G + B);
            }
            
            // Clamp values
            R = max(0.0, min(1.0, R));
            G = max(0.0, min(1.0, G));
            B = max(0.0, min(1.0, B));
            
            rgbImage.at<Vec3d>(i, j) = Vec3d(B, G, R);
        }
    }
    
    Mat result;
    rgbImage.convertTo(result, CV_8UC3, 255.0);
    return result;
}

// ============================================================================
// CONVERSIE XYZ
// ============================================================================

Mat rgb2xyz(const Mat& rgbImage) {
    Mat rgbFloat;
    rgbImage.convertTo(rgbFloat, CV_64FC3, 1.0/255.0);
    
    Mat xyzImage(rgbFloat.size(), CV_64FC3);
    
    // Matricea de transformare sRGB -> XYZ (D65 illuminant)
    for (int i = 0; i < rgbFloat.rows; i++) {
        for (int j = 0; j < rgbFloat.cols; j++) {
            Vec3d rgb = rgbFloat.at<Vec3d>(i, j);
            double B = rgb[0], G = rgb[1], R = rgb[2];
            
            // Aplicare gamma correction (sRGB la linear)
            auto gammaCorrect = [](double v) {
                return (v > 0.04045) ? pow((v + 0.055) / 1.055, 2.4) : v / 12.92;
            };
            
            R = gammaCorrect(R);
            G = gammaCorrect(G);
            B = gammaCorrect(B);
            
            // Transformare liniară
            double X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
            double Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
            double Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B;
            
            // Scalare pentru D65
            X *= 100;
            Y *= 100;
            Z *= 100;
            
            xyzImage.at<Vec3d>(i, j) = Vec3d(X, Y, Z);
        }
    }
    
    return xyzImage;
}

Mat xyz2rgb(const Mat& xyzImage) {
    Mat rgbImage(xyzImage.size(), CV_64FC3);
    
    for (int i = 0; i < xyzImage.rows; i++) {
        for (int j = 0; j < xyzImage.cols; j++) {
            Vec3d xyz = xyzImage.at<Vec3d>(i, j);
            double X = xyz[0] / 100.0;
            double Y = xyz[1] / 100.0;
            double Z = xyz[2] / 100.0;
            
            // Matricea inversă XYZ -> sRGB
            double R =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
            double G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
            double B =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
            
            // Gamma correction inversă
            auto gammaInverse = [](double v) {
                return (v > 0.0031308) ? 1.055 * pow(v, 1.0/2.4) - 0.055 : 12.92 * v;
            };
            
            R = gammaInverse(R);
            G = gammaInverse(G);
            B = gammaInverse(B);
            
            // Clamp values
            R = max(0.0, min(1.0, R));
            G = max(0.0, min(1.0, G));
            B = max(0.0, min(1.0, B));
            
            rgbImage.at<Vec3d>(i, j) = Vec3d(B, G, R);
        }
    }
    
    Mat result;
    rgbImage.convertTo(result, CV_8UC3, 255.0);
    return result;
}

// ============================================================================
// CONVERSIE CIELAB
// ============================================================================

double labFunction(double q) {
    // Formula (23)
    if (q > 0.008856) {
        return pow(q, 1.0/3.0);
    } else {
        return 7.787 * q + 16.0/116.0;
    }
}

double labFunctionInverse(double q) {
    double q3 = q * q * q;
    if (q3 > 0.008856) {
        return q3;
    } else {
        return (q - 16.0/116.0) / 7.787;
    }
}

Mat rgb2lab(const Mat& rgbImage) {
    // Mai întâi convertim la XYZ
    Mat xyzImage = rgb2xyz(rgbImage);
    
    Mat labImage(xyzImage.size(), CV_64FC3);
    
    for (int i = 0; i < xyzImage.rows; i++) {
        for (int j = 0; j < xyzImage.cols; j++) {
            Vec3d xyz = xyzImage.at<Vec3d>(i, j);
            double X = xyz[0];
            double Y = xyz[1];
            double Z = xyz[2];
            
            // Formulas (20), (21), (22)
            double L = 116.0 * labFunction(Y / Y0) - 16.0;
            double a = 500.0 * (labFunction(X / X0) - labFunction(Y / Y0));
            double b = 200.0 * (labFunction(Y / Y0) - labFunction(Z / Z0));
            
            labImage.at<Vec3d>(i, j) = Vec3d(L, a, b);
        }
    }
    
    return labImage;
}

Mat lab2rgb(const Mat& labImage) {
    Mat xyzImage(labImage.size(), CV_64FC3);
    
    for (int i = 0; i < labImage.rows; i++) {
        for (int j = 0; j < labImage.cols; j++) {
            Vec3d lab = labImage.at<Vec3d>(i, j);
            double L = lab[0];
            double a = lab[1];
            double b = lab[2];
            
            double fy = (L + 16.0) / 116.0;
            double fx = a / 500.0 + fy;
            double fz = fy - b / 200.0;
            
            double X = X0 * labFunctionInverse(fx);
            double Y = Y0 * labFunctionInverse(fy);
            double Z = Z0 * labFunctionInverse(fz);
            
            xyzImage.at<Vec3d>(i, j) = Vec3d(X, Y, Z);
        }
    }
    
    return xyz2rgb(xyzImage);
}

// ============================================================================
// APLICAȚII DE PROCESARE
// ============================================================================

Mat equalizeHistogramRGB(const Mat& rgbImage) {
    vector<Mat> channels;
    split(rgbImage, channels);
    
    for (int i = 0; i < 3; i++) {
        equalizeHist(channels[i], channels[i]);
    }
    
    Mat result;
    merge(channels, result);
    return result;
}

Mat equalizeHistogramHSI(const Mat& rgbImage) {
    // Convertim la HSI
    Mat hsiImage = rgb2hsi(rgbImage);
    
    // Separăm canalele
    vector<Mat> channels;
    split(hsiImage, channels);
    
    // Egalizăm doar componenta I (Intensity)
    Mat I_channel;
    channels[2].convertTo(I_channel, CV_8U, 255.0);
    equalizeHist(I_channel, I_channel);
    I_channel.convertTo(channels[2], CV_64F, 1.0/255.0);
    
    // Reunim canalele
    Mat hsiEqualized;
    merge(channels, hsiEqualized);
    
    // Convertim înapoi la RGB
    return hsi2rgb(hsiEqualized);
}

Mat smoothingRGB(const Mat& rgbImage, int kernelSize) {
    Mat result;
    blur(rgbImage, result, Size(kernelSize, kernelSize));
    return result;
}

Mat smoothingHSI(const Mat& rgbImage, int kernelSize) {
    // Convertim la HSI
    Mat hsiImage = rgb2hsi(rgbImage);
    
    // Separăm canalele
    vector<Mat> channels;
    split(hsiImage, channels);
    
    // Aplicăm netezire doar pe componenta I
    blur(channels[2], channels[2], Size(kernelSize, kernelSize));
    
    // Reunim canalele
    Mat hsiSmoothed;
    merge(channels, hsiSmoothed);
    
    // Convertim înapoi la RGB
    return hsi2rgb(hsiSmoothed);
}

Mat sharpeningLaplacianRGB(const Mat& rgbImage) {
    vector<Mat> channels;
    split(rgbImage, channels);
    
    // Kernel Laplacian
    Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    
    for (int i = 0; i < 3; i++) {
        Mat channelFloat;
        channels[i].convertTo(channelFloat, CV_32F);
        filter2D(channelFloat, channelFloat, -1, kernel);
        channelFloat.convertTo(channels[i], CV_8U);
    }
    
    Mat result;
    merge(channels, result);
    return result;
}

// ============================================================================
// FUNCȚII AUXILIARE
// ============================================================================

void visualizeColorChannels(const Mat& image, const string& spaceName, 
                            const vector<string>& channelNames, const string& outputDir) {
    vector<Mat> channels;
    split(image, channels);
    
    cout << "  Vizualizare spațiu " << spaceName << ":" << endl;
    
    for (size_t i = 0; i < channels.size() && i < channelNames.size(); i++) {
        Mat display;
        
        if (channels[i].type() == CV_64F) {
            // Normalizare pentru afișare
            double minVal, maxVal;
            minMaxLoc(channels[i], &minVal, &maxVal);
            channels[i].convertTo(display, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        } else {
            display = channels[i].clone();
        }
        
        string filename = outputDir + "/" + spaceName + "_" + channelNames[i] + ".png";
        imwrite(filename, display);
        cout << "    ✓ Canal " << channelNames[i] << " salvat: " << filename << endl;
    }
}

void saveImage(const Mat& image, const string& filename) {
    imwrite(filename, image);
    cout << "✓ Salvat: " << filename << endl;
}
