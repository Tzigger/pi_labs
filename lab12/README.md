# Laborator 12 - Segmentarea Imaginilor cu Watershed

## Descriere
Acest laborator implementează **algoritmul Watershed** pentru segmentarea imaginilor, urmând metodologia în 9 pași:

1. **Segmentare grosieră** cu threshold Otsu
2. **Corectare morfologică** (închidere + deschidere)
3. **Identificare background cert** prin dilatare
4. **Transformată distanță** cu DIST_L2
5. **Identificare foreground cert** prin threshold pe distanță
6. **Separare obiecte** cu connectedComponents
7. **Regiuni de incertitudine** (background - foreground)
8. **Construire markeri** pentru watershed
9. **Aplicare watershed** și marcare contururi

## Algoritmul Watershed

Watershed ("linia de cumpănă a apelor") este un algoritm de segmentare bazat pe analogia cu inundarea topografică:
- Imaginea este tratată ca o suprafață topografică
- Intensitățile pixelilor reprezintă altitudini
- Algoritmul "inundă" progresiv bazinele pornind de la markeri
- Când două bazine se întâlnesc, se creează o "linie de cumpănă" (contur)

### Avantaje:
- Segmentare precisă a obiectelor care se ating
- Controlabil prin markeri (semi-automat)
- Funcționează bine pentru obiecte convexe

### Dezavantaje:
- Sensibil la zgomot (necesită preprocesare)
- Poate produce over-segmentation
- Necesită markeri de calitate

## Exerciții implementate

### Exercițiul 1: Segmentare Otsu
Segmentare grosieră folosind metoda Otsu pentru determinarea automată a pragului.

**Probleme observate:**
- Cavități în interiorul obiectelor
- Puncte singulare în exterior

### Exercițiul 2: Corectare morfologică
Aplicarea operațiilor morfologice pentru corectarea erorilor:
- **Închidere** (dilate + erode): elimină cavitățile interioare
- **Deschidere** (erode + dilate): elimină punctele exterioare

**Element structural:** kernel dreptunghiular 5x5

### Exercițiul 3: Background cert
Dilatare morfologică pentru exagerarea contururilor obiectelor.

**Rezultat:** Regiunile negre = background cert, regiunile albe = posibile obiecte sau incertitudine

**Element structural:** cerc (MORPH_ELLIPSE) 3x3

### Exercițiul 4: Transformată distanță
Pentru fiecare pixel se calculează distanța până la cel mai apropiat pixel de valoare 0.

**Funcție:** `distanceTransform(image, output, DIST_L2, 5)`
- `DIST_L2`: distanță Euclidiană
- Rezultat: matrice de float-uri cu distanțele

### Exercițiul 5: Foreground cert
Segmentare cu prag pe transformata distanță pentru a identifica centrele obiectelor.

**Prag:** 70% din valoarea maximă a distanței (ajustabil)

**Rezultat:** Pixeli albi = cu certitudine interior obiect

### Exercițiul 6: Connected components
Separarea regiunilor identificate pe obiecte individuale.

**Funcție:** `connectedComponents(foreground, labels)`
- Background = 0
- Obiecte = 1, 2, 3, ... (în ordine crescătoare)

### Exercițiul 7: Regiuni de incertitudine
Identificarea zonelor unde algoritmul nu este sigur dacă aparțin obiectelor sau background-ului.

**Formula:** `uncertain = sureBackground - sureForeground`

### Exercițiul 8: Construire markeri
Construirea matricei de markeri necesară pentru watershed:
1. Adaugă 1 la toate label-urile (background 0→1, obiecte 1,2,3→2,3,4)
2. Setează pe 0 regiunile de incertitudine

**Rezultat:**
- Incertitudine = 0
- Background = 1
- Centre obiecte = 2, 3, 4, ...

### Exercițiul 9: Aplicare watershed
Aplicarea algoritmului watershed pe imaginea RGB cu markerii construiți.

**Funcție:** `watershed(imageRGB, markers)`
- Modifică matricea markers
- Setează pe -1 contururile (boundaries)

**Vizualizare:** Contururile sunt marcate în roșu pe imaginea originală

### Exercițiul 10: Rulare completă
Rulează automat toți pașii 1-9 și salvează toate rezultatele intermediare.

**Output:** 11 fișiere PNG în `output_lab12/`

## Compilare și rulare

### Metoda 1: Script automat
```bash
cd lab12
./run_lab12.sh
```

### Metoda 2: Manual
```bash
cd lab12
cmake .
make
./lab12 Imagini_Laborator/coins.jpg
```

## Exemple de utilizare

### Cu imaginea coins.jpg (recomandată)
```bash
./lab12 Imagini_Laborator/coins.jpg
# Selectează opțiunea 10 pentru rulare completă
```

### Pași individuali
```bash
./lab12 Imagini_Laborator/coins.jpg
# Selectează 1-9 pentru fiecare pas individual
# Selectează 10 pentru rulare automată
```

## Parametri ajustabili

În fișierul `watershed.cpp` puteți modifica:

- **Kernel morfologic (Pasul 2):**
  ```cpp
  Mat corrected = step2_morphologicalCorrection(binary, 5); // dimensiune 5x5
  ```

- **Dilatare background (Pasul 3):**
  ```cpp
  Mat sureBg = step3_identifySureBackground(corrected, 3); // rază 3
  ```

- **Threshold distanță (Pasul 5):**
  ```cpp
  Mat sureFg = step5_identifySureForeground(distTransform, 0.7); // 70% din max
  ```

## Fișiere output

Toate rezultatele sunt salvate în `output_lab12/`:

| Fișier | Descriere |
|--------|-----------|
| `step1_otsu_threshold.png` | Segmentare Otsu inițială (obiecte albe) |
| `step2_morphological_correction.png` | După corectare morfologică |
| `step3_sure_background.png` | Background cert (dilatat) |
| `step4_distance_transform.png` | Transformată distanță (normalizată) |
| `step5_sure_foreground.png` | Foreground cert (centre obiecte) |
| `step6_connected_components.png` | Label-uri componente conexe |
| `step7_uncertain_region.png` | Regiuni de incertitudine |
| `step8_markers.png` | Markeri pentru watershed (grayscale) |
| `step8_markers_color.png` | **Markeri colorați (albastru=incertitudine)** |
| `step9_watershed_result.png` | Rezultat final cu contururi |
| `step9_markers_final.png` | Markeri după watershed |

## Structura proiectului

```
lab12/
├── main_lab12.cpp          # Programul principal cu meniu
├── watershed.h             # Header cu declarații funcții
├── watershed.cpp           # Implementare algoritm watershed
├── CMakeLists.txt          # Configurare build
├── run_lab12.sh            # Script rulare automată
├── README.md               # Acest fișier
├── Imagini_Laborator/      # Imagini de test
│   ├── coins.jpg          # Monede (recomandat)
│   └── weld.jpg           # Sudură
└── output_lab12/           # Rezultate (generat automat)
```

## Dependențe

- **OpenCV 4.x** (testat cu 4.12.0)
- **C++17** (pentru std::filesystem)
- **CMake 3.10+**

## Funcții OpenCV utilizate

| Funcție | Scop |
|---------|------|
| `threshold()` | Segmentare cu prag, Otsu |
| `morphologyEx()` | Operații morfologice (MORPH_CLOSE, MORPH_OPEN) |
| `dilate()` | Dilatare morfologică |
| `distanceTransform()` | Calculare transformată distanță |
| `connectedComponents()` | Etichetare componente conexe |
| `watershed()` | Algoritmul watershed |
| `setTo()` | Setare valori condiționat |
| `cvtColor()` | Conversie color (GRAY2BGR) |

## Note importante

1. **Ordinea operațiilor morfologice:**
   - Închidere (dilate→erode) elimină găuri mici
   - Deschidere (erode→dilate) elimină puncte izolate

2. **Transformata distanță:**
   - DIST_L2 = distanță Euclidiană (mai precisă)
   - Valorile mari = pixeli departe de margini (centre obiecte)

3. **Markeri watershed:**
   - Trebuie să fie CV_32S (întregi pe 32 biți)
   - Valoarea 0 = regiuni necunoscute (vor fi segmentate)
   - Valori pozitive = regiuni cunoscute (markeri)
   - Valoarea -1 = contururi (după watershed)

4. **Threshold distanță:**
   - Valori mai mari (0.8-0.9) = regiuni mai mici, mai sigure
   - Valori mai mici (0.5-0.7) = regiuni mai mari, posibil over-segmentation

## Debugging

Pentru a vedea valorile intermediate:
```bash
# Rulează un pas individual și verifică output-ul
./lab12 Imagini_Laborator/coins.jpg
# Selectează opțiunea dorită (1-9)
# Imaginile vor fi afișate și salvate
```

## Exemple de rezultate

### Coins.png
- **Obiecte detectate:** ~10 monede
- **Prag Otsu:** ~130
- **Componente conexe:** ~10 obiecte
- **Pixeli contur:** ~1000-2000

### Weld.jpg
- **Obiecte detectate:** zone de sudură
- **Prag Otsu:** ~150
- **Rezultat:** segmentare defecte sudură

## Referințe

- OpenCV Documentation: [Watershed Algorithm](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
- Distance Transform: [cv::distanceTransform](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042)
- Morphological Operations: [cv::morphologyEx](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
