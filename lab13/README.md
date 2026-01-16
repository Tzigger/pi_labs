# Laborator 13 - Modele (Spații) de Culoare

## Descriere

Acest laborator implementează conversii între diferite spații de culoare și aplicații de procesare a imaginilor color:

### Modele de culoare implementate:
1. **RGB** - Red, Green, Blue (dispozitive de afișare)
2. **CMY(K)** - Cyan, Magenta, Yellow, blacK (dispozitive de imprimare)
3. **YIQ** - Luminanță și crominanță (NTSC TV)
4. **HSI** - Hue, Saturation, Intensity (percepție umană)
5. **XYZ** - CIE 1931 (spațiu colorimetric)
6. **CIELAB** - L*a*b* (uniform perceptual)

### Aplicații practice:
1. **Egalizare histogramă** - comparație RGB vs HSI
2. **Netezire (filtru medie)** - comparație RGB vs HSI cu dimensiuni 5, 10, 25
3. **Accentuare margini** - filtru Laplacian în RGB

## Teoria

### 1. Modelul RGB
- Spațiu aditiv de culoare, cub unitate [0,1]³
- Culorile primare: Red, Green, Blue
- Dependent de dispozitiv

### 2. Modelul CMY(K)
Conversie RGB → CMY:
```
C = 1 - R
M = 1 - G  
Y = 1 - B
```

Pentru CMYK:
```
K = min(C, M, Y)
C = C - K, M = M - K, Y = Y - K
```

### 3. Modelul YIQ (NTSC)
Matricea de transformare:
```
| Y |   | 0.299   0.587   0.114 | | R |
| I | = | 0.596  -0.274  -0.322 | | G |
| Q |   | 0.211  -0.523   0.312 | | B |
```

### 4. Modelul HSI
- **H (Hue)**: nuanța culorii (0-360°)
- **S (Saturation)**: puritatea culorii (0-1)
- **I (Intensity)**: intensitatea luminoasă (0-1)

Formule:
```
I = (R + G + B) / 3
S = 1 - 3·min(R,G,B) / (R+G+B)
H = arccos[(R-G)+(R-B) / 2√((R-G)² + (R-B)(G-B))]
```

### 5. Spațiul XYZ
Transformare liniară a RGB cu corecție gamma, astfel încât toate culorile fizice au coordonate pozitive.

### 6. Sistemul CIELAB
Spațiu uniform perceptual:
- **L*** - luminozitate (0=negru, 100=alb)
- **a*** - axa roșu-verde
- **b*** - axa galben-albastru

Formule:
```
L* = 116·h(Y/Y₀) - 16
a* = 500·[h(X/X₀) - h(Y/Y₀)]
b* = 200·[h(Y/Y₀) - h(Z/Z₀)]
```
unde h(q) = q^(1/3) dacă q > 0.008856, altfel 7.787·q + 16/116

## Compilare și rulare

### Metoda 1: Script automat
```bash
cd lab13
chmod +x run_lab13.sh
./run_lab13.sh
```

### Metoda 2: Manual
```bash
cd lab13
cmake .
make
./lab13 Imagini_Laborator/lena.png
```

## Meniu interactiv

```
====== LABORATOR 13 - Modele de Culoare ======
1.  Vizualizare spații de culoare
2.  Conversie RGB -> CMY și înapoi
3.  Conversie RGB -> CMYK și înapoi
4.  Conversie RGB -> YIQ și înapoi
5.  Conversie RGB -> HSI și înapoi
6.  Conversie RGB -> XYZ și înapoi
7.  Conversie RGB -> LAB și înapoi
----------------------------------------------
8.  Aplicația 1: Egalizare histogramă (RGB vs HSI)
9.  Aplicația 2: Netezire filtru medie (RGB vs HSI)
10. Aplicația 3: Accentuare Laplacian (RGB)
----------------------------------------------
11. Rulează TOATE aplicațiile și salvează
0.  Ieșire
==============================================
```

## Aplicații detaliate

### Aplicația 1: Egalizare histogramă
- **RGB**: Egalizare pe fiecare canal separat → poate introduce artefacte de culoare
- **HSI**: Egalizare doar pe componenta I → păstrează nuanțele originale

### Aplicația 2: Netezire
Comparație între netezire în RGB vs HSI pentru:
- Kernel 5×5
- Kernel 10×10  
- Kernel 25×25

**Observații**:
- RGB: amestecă culorile la margini
- HSI: păstrează nuanțele, doar intensitatea se netezește

### Aplicația 3: Accentuare Laplacian
Kernel folosit:
```
| 0  -1   0 |
|-1   5  -1 |
| 0  -1   0 |
```

## Structura proiectului

```
lab13/
├── main_lab13.cpp         # Program principal cu meniu
├── color_spaces.h         # Header funcții conversie
├── color_spaces.cpp       # Implementare conversii
├── CMakeLists.txt         # Configurare build
├── run_lab13.sh           # Script rulare
├── README.md              # Acest fișier
├── Imagini_Laborator/     # Imagini test
└── output_lab13/          # Rezultate (generat automat)
```

## Fișiere output

| Fișier | Descriere |
|--------|-----------|
| `RGB_R/G/B.png` | Canale RGB separate |
| `HSI_H/S/I.png` | Componente HSI |
| `CMY_C/M/Y.png` | Componente CMY |
| `YIQ_Y/I/Q.png` | Componente YIQ |
| `App1_equalized_RGB.png` | Egalizare pe canale RGB |
| `App1_equalized_HSI.png` | Egalizare pe componenta I |
| `App2_smooth_RGB_5/10/25.png` | Netezire RGB |
| `App2_smooth_HSI_5/10/25.png` | Netezire HSI |
| `App3_sharpened_Laplacian.png` | Accentuare margini |

## Dependențe

- **OpenCV 4.x** (testat cu 4.12.0)
- **C++17** (pentru std::filesystem)
- **CMake 3.10+**

## Funcții OpenCV utilizate

| Funcție | Scop |
|---------|------|
| `imread()` | Citire imagine |
| `imwrite()` | Salvare imagine |
| `split()` / `merge()` | Separare/reunire canale |
| `equalizeHist()` | Egalizare histogramă |
| `blur()` | Filtru medie |
| `filter2D()` | Convoluție cu kernel |
| `Laplacian()` | Operator Laplacian |
| `cvtColor()` | Conversie spații de culoare |

## Referințe

- OpenCV Color Conversions: [cv::cvtColor](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- Histogram Equalization: [cv::equalizeHist](https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html)
- Image Filtering: [cv::blur, cv::filter2D](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)
