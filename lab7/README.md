# Laborator 7 - Prelucrarea Imaginilor

## Descriere
Laborator 7 implementează operații de prelucrare a imaginilor, incluzând:
- Negarea imaginilor
- Binarizarea imaginilor
- Egalizarea histogramelor
- Filtre spațiale (Sobel, Gaussian Blur, Motion Blur, etc.)
- Detectarea muchiilor

## Structura Proiectului
```
lab7/
├── main.cpp                      # Program principal
├── main_save_files.cpp           # Program pentru salvarea fișierelor
├── image_processing_helpers.h    # Header cu funcții de prelucrare
├── CMakeLists.txt                # Configurație CMake
├── run_lab.sh                    # Script de rulare
├── run_gui.sh                    # Script GUI
├── Imagini_Laborator/            # Imagini de test
└── output/                       # Directorul pentru rezultate
```

## Compilare și Rulare

### Metoda 1: Folosind scriptul
```bash
chmod +x run_lab.sh
./run_lab.sh
```

### Metoda 2: Manual
```bash
cmake .
make
./lab7
```

### Metoda 3: GUI
```bash
chmod +x run_gui.sh
./run_gui.sh
```

## Funcționalități
1. Negarea imaginilor
2. Binarizare cu prag personalizat
3. Egalizarea histogramei
4. Filtre de netezire (Gaussian, Mean, Median, Motion Blur)
5. Detectarea muchiilor (Sobel, Laplacian)
6. Combinarea muchiilor orizontale și verticale

## Cerințe
- OpenCV 4.x
- CMake 3.10+
- C++17
