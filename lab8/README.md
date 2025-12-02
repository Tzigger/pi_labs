# Laborator 8 - Filtrarea Imaginilor în Domeniul Spațial

## Descriere
Laborator 8 se concentrează pe filtrarea imaginilor în domeniul spațial folosind convoluție:
- Filtre de netezire (Low-pass)
- Filtre de accentuare (High-pass)
- Filtre personalizate
- Analiza și afișarea filtrelor

## Structura Proiectului
```
lab8/
├── main_lab8.cpp                 # Program principal
├── image_processing_helpers.h    # Header cu funcții de prelucrare
├── Complex.h                     # Funcții pentru numere complexe
├── CMakeLists.txt                # Configurație CMake
├── run_lab8.sh                   # Script de rulare
├── Imagini_Laborator/            # Imagini de test
└── output_lab8/                  # Directorul pentru rezultate
```

## Compilare și Rulare

### Metoda 1: Folosind scriptul
```bash
chmod +x run_lab8.sh
./run_lab8.sh
```

### Metoda 2: Manual
```bash
cmake .
make
./lab8
```

## Funcționalități

### Filtre de Netezire
- Filtru medie aritmetică (Mean Filter)
- Filtru Gaussian
- Filtru median
- Motion Blur

### Filtre de Accentuare
- Filtru Laplacian
- Filtru Sobel (detectare muchii)
- High-pass filter personalizat

### Operații Suplimentare
- Afișarea kernel-urilor de convoluție
- Compararea rezultatelor diferitelor filtre
- Salvarea rezultatelor

## Meniu Interactiv
```
1. Aplicare filtru de netezire
2. Aplicare filtru de accentuare
3. Comparare filtre de netezire
4. Comparare filtre de accentuare
5. Afișare kernel-uri
0. Ieșire
```

## Cerințe
- OpenCV 4.x
- CMake 3.10+
- C++17

## Parametri Personalizabili
- Dimensiunea kernel-ului (3x3, 5x5, 7x7, etc.)
- Sigma pentru filtrul Gaussian
- Direcția pentru Motion Blur
