# Laboratoare Prelucrarea Imaginilor

Acest repository conține implementările pentru laboratoarele 7, 8 și 9 de la cursul de Prelucrarea Imaginilor.

## Structura Proiectului

```
PI/
├── lab7/                          # Laborator 7 - Prelucrarea imaginilor (operații de bază)
│   ├── main.cpp
│   ├── main_save_files.cpp
│   ├── image_processing_helpers.h
│   ├── CMakeLists.txt
│   ├── run_lab.sh
│   ├── README.md
│   ├── Imagini_Laborator/
│   └── output/
│
├── lab8/                          # Laborator 8 - Filtrarea în domeniul spațial
│   ├── main_lab8.cpp
│   ├── image_processing_helpers.h
│   ├── Complex.h
│   ├── CMakeLists.txt
│   ├── run_lab8.sh
│   ├── README.md
│   ├── Imagini_Laborator/
│   └── output_lab8/
│
└── lab9/                          # Laborator 9 - Domeniul frecvențelor (Fourier)
    ├── main_lab9.cpp
    ├── frequency_domain_helpers.h
    ├── frequency_domain_helpers.cpp
    ├── CMakeLists.txt
    ├── run_lab9.sh
    ├── README.md
    ├── Imagini_Laborator/
    └── output_lab9/
```

## Cerințe Generale

- **OpenCV 4.x**: Bibliotecă pentru prelucrarea imaginilor
- **CMake 3.10+**: Sistem de build
- **C++17**: Standard C++ utilizat
- **macOS/Linux**: Sisteme de operare suportate

### Instalare OpenCV (macOS)
```bash
brew install opencv
```

### Instalare OpenCV (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

## Laboratoare

### Laborator 7 - Prelucrarea Imaginilor (Operații de Bază)
**Locație**: `lab7/`

**Funcționalități**:
- Negarea imaginilor
- Binarizarea cu prag
- Egalizarea histogramelor
- Filtre spațiale de bază
- Detectarea muchiilor

**Rulare**:
```bash
cd lab7
./run_lab.sh
# SAU
cmake . && make && ./lab7
```

[Documentație completă →](lab7/README.md)

---

### Laborator 8 - Filtrarea în Domeniul Spațial
**Locație**: `lab8/`

**Funcționalități**:
- Filtre de netezire (Mean, Gaussian, Median, Motion Blur)
- Filtre de accentuare (Laplacian, Sobel, High-pass)
- Convoluție personalizată
- Compararea diferitelor filtre

**Rulare**:
```bash
cd lab8
./run_lab8.sh
# SAU
cmake . && make && ./lab8
```

[Documentație completă →](lab8/README.md)

---

### Laborator 9 - Prelucrarea în Domeniul Frecvențelor
**Locație**: `lab9/`

**Funcționalități**:
- Transformata Fourier Discretă (DFT)
- Transformata Fourier Inversă (IDFT)
- Filtre trece-jos (Ideal, Butterworth, Gaussian)
- Filtre trece-sus (Ideal, Butterworth, Gaussian)
- Afișarea spectrului Fourier
- Compararea diferitelor tipuri de filtre

**Rulare**:
```bash
cd lab9
./run_lab9.sh
# SAU
cmake . && make && ./lab9
```

[Documentație completă →](lab9/README.md)

---

## Compilare Rapidă

Fiecare laborator poate fi compilat independent:

```bash
# Laborator 7
cd lab7 && cmake . && make

# Laborator 8
cd lab8 && cmake . && make

# Laborator 9
cd lab9 && cmake . && make
```

## Utilizare

Toate laboratoarele au meniuri interactive. Rulați executabilul și urmați instrucțiunile:

```bash
# Exemplu pentru lab9
cd lab9
./lab9

# Cu imagine personalizată
./lab9 path/to/image.pgm
```

## Imagini de Test

Fiecare laborator conține directorul `Imagini_Laborator/` cu imagini de test:
- `house.pgm` - Imagine principală de test
- Alte imagini specifice fiecărui laborator

## Rezultate

Fiecare laborator salvează rezultatele în directorul propriu:
- Lab 7: `lab7/output/`
- Lab 8: `lab8/output_lab8/`
- Lab 9: `lab9/output_lab9/`

## Funcții OpenCV Utilizate

### Laborator 7
- `cv::imread`, `cv::imwrite`, `cv::imshow`
- `cv::cvtColor`
- `cv::equalizeHist`
- `cv::GaussianBlur`, `cv::medianBlur`
- `cv::Sobel`, `cv::Laplacian`

### Laborator 8
- `cv::filter2D` - Convoluție personalizată
- `cv::blur` - Mean filter
- `cv::GaussianBlur` - Gaussian filter
- `cv::medianBlur` - Median filter
- `cv::Sobel`, `cv::Laplacian` - Detectare muchii

### Laborator 9
- `cv::dft` - Transformata Fourier Discretă
- `cv::idft` - Transformata Fourier Inversă
- `cv::magnitude` - Magnitudine complexă
- `cv::log` - Scalare logaritmică
- `cv::normalize` - Normalizare
- `cv::split`, `cv::merge` - Manipulare canale
- `cv::multiply` - Multiplicare element cu element

## Contribuții

Fiecare laborator este independent și poate fi modificat/extins separat.

## Licență

Proiect educațional pentru cursul de Prelucrarea Imaginilor.

## Autor

Realizat pentru laboratoarele de Prelucrarea Imaginilor (PI)

---

## Quick Start

```bash
# Clonează repository-ul
git clone [repository-url]
cd PI

# Alege un laborator
cd lab9

# Compilează și rulează
./run_lab9.sh
```

Pentru informații detaliate despre fiecare laborator, consultați fișierele README.md din fiecare director.
