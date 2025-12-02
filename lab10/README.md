# Laborator 10 - Segmentarea Imaginilor (1)

## Descriere

Acest laborator implementează tehnici de segmentare a imaginilor bazate pe delimitarea cu prag (thresholding).

## Concepte Teoretice

### Segmentarea imaginilor
Scopul segmentării este de a separa obiectele dintr-o imagine de fundal. Algoritmii se bazează pe:
- **Discontinuitate**: variații bruste ale intensității (detectare muchii)
- **Similaritate**: regiuni cu caracteristici asemănătoare (segmentare cu prag)

### Tipuri de praguri

1. **Prag global simplu**:
   ```
   g(x,y) = 255 dacă f(x,y) >= T, altfel 0
   ```

2. **Prag de bandă**:
   ```
   g(x,y) = 255 dacă T1 <= f(x,y) <= T2, altfel 0
   ```

3. **Praguri multiple**:
   - Împarte imaginea în mai multe regiuni bazate pe mai multe praguri

4. **Semiprag**:
   ```
   g(x,y) = f(x,y) dacă f(x,y) >= T, altfel 0
   ```
   (mascarea fundalului - păstrează valorile originale pentru obiekte)

### Metode de selectare a pragului

1. **Metoda Otsu**: Minimizează varianța intra-clasă
2. **Metoda Iterativă**: Calculează iterativ pragul până la convergență
3. **Analiza histogramei**: Găsește minimele locale în histogramă

## Funcționalități implementate

| Nr. | Funcție | Descriere |
|-----|---------|-----------|
| 1 | Prag global | Binarizare simplă cu prag fix |
| 2 | Prag de bandă | Selectează pixeli într-un interval |
| 3 | Praguri multiple | Creează mai multe regiuni |
| 4 | Semiprag | Mascarea fundalului |
| 5 | Otsu | Selectare automată optimală |
| 6 | Iterativ | Selectare automată prin iterații |
| 7 | Adaptiv | Prag local variabil |
| 8 | Comparație | Toate metodele simultan |
| 9 | Histogramă | Vizualizare histogramă |

## Compilare și rulare

```bash
# Compilare
chmod +x run_lab10.sh
./run_lab10.sh

# Sau manual
cmake .
make
./lab10 cale/catre/imagine.png
```

## Structura fișierelor

```
lab10/
├── main_lab10.cpp           # Programul principal
├── segmentation_helpers.h   # Header funcții segmentare
├── segmentation_helpers.cpp # Implementare funcții
├── CMakeLists.txt           # Configurare CMake
├── run_lab10.sh             # Script compilare/rulare
├── README.md                # Documentație
└── output_lab10/            # Rezultate (generat automat)
```

## Exemple de utilizare

### 1. Prag global
```cpp
Mat segmented = applyGlobalThreshold(image, 128);
```

### 2. Metoda Otsu
```cpp
int threshold = computeOtsuThreshold(image);
Mat binary = applyGlobalThreshold(image, threshold);
```

### 3. Segmentare adaptivă
```cpp
Mat adaptive = applyAdaptiveThreshold(image, 21, 5, 0);
```

## Output

Rezultatele sunt salvate în directorul `output_lab10/`:
- `ex1_threshold_*.png` - Rezultate prag global
- `ex2_band_*.png` - Rezultate prag de bandă
- `ex3_multiple_*.png` - Rezultate praguri multiple
- `ex4_semi_*.png` - Rezultate semiprag
- `ex5_otsu.png` - Rezultate metoda Otsu
- `ex6_iterative.png` - Rezultate metoda iterativă
- `ex7_adaptive_*.png` - Rezultate segmentare adaptivă
- `ex8_all_methods.png` - Comparație toate metodele
- `ex9_histogram.png` - Histograma imaginii
