# Laborator 11 - Segmentarea Imaginilor (2) - Region Growing

## Descriere
Acest laborator implementează algoritmul de **Region Growing** pentru segmentarea imaginilor, folosind trei abordări diferite:
1. **Recursiv 4-conectat** - consideră doar vecinii orizontali și verticali
2. **Recursiv 8-conectat** - include și vecinii diagonali
3. **cv::floodFill** - folosește funcția OpenCV optimizată

## Exerciții implementate

### Exercițiul 1: Region Growing Recursiv (4-conectat)
Implementare recursivă care explorează doar cei 4 vecini (stânga, dreapta, sus, jos).

### Exercițiul 2: Region Growing Recursiv (8-conectat)
Implementare recursivă extinsă cu toți cei 8 vecini (include diagonalele).

### Exercițiul 3: Region Growing cu cv::floodFill
Utilizează funcția optimizată OpenCV cu flag `FLOODFILL_FIXED_RANGE` pentru compararea cu seed-ul original.

### Exercițiul 4: Multi-seed Region Growing
Aplică region growing pe multiple puncte seed simultan, cu threshold-uri diferite pentru fiecare seed.

### Exercițiul 5: Comparație toate metodele
Afișează side-by-side rezultatele tuturor metodelor pentru același seed point.

### Exercițiul 6: Aplicație interactivă
- Click pe imagine pentru a selecta puncte seed
- `+/-` pentru ajustarea threshold-ului
- `r` pentru reset
- `s` pentru salvare
- `ESC` pentru ieșire

### Exercițiul 7: Rulare automată
Rulează automat toate exercițiile și salvează rezultatele în `output_lab11/`.

## Compilare și rulare

### Metoda 1: Script automat
```bash
./run_lab11.sh
```

### Metoda 2: Manual
```bash
cmake .
make
./lab11 <cale_imagine>
```

## Exemple de utilizare

### Cu imaginea weld.jpg (recomandată)
```bash
./lab11 Imagini_Laborator/weld.jpg
```

### Cu imaginea coins
```bash
./lab11 Imagini_Laborator/coins.png
```

## Parametri seed predefiniti (Ex. 4)
Conform exemplului din laborator:
- **P1**: x=120, y=218, seedValue=255, threshold=40 (roșu)
- **P2**: x=250, y=215, seedValue=255, threshold=80 (verde)
- **P3**: x=380, y=210, seedValue=255, threshold=110 (albastru)

## Fișiere output
Toate rezultatele sunt salvate în `output_lab11/`:
- `ex1_recursive_4connected.png` - Masca 4-conectat
- `ex2_recursive_8connected.png` - Masca 8-conectat
- `ex3_floodfill.png` - Masca floodFill
- `ex4_multiseed_*.png` - Rezultate multi-seed
- `ex5_comparison_all_methods.png` - Comparație metode
- `ex6_interactive_result.png` - Salvare din modul interactiv

## Structura proiectului
```
lab11/
├── main_lab11.cpp          # Programul principal
├── region_growing.h        # Header cu declarații
├── region_growing.cpp      # Implementare funcții
├── CMakeLists.txt          # Configurare build
├── run_lab11.sh            # Script rulare
├── README.md               # Acest fișier
├── Imagini_Laborator/      # Imagini de test
└── output_lab11/           # Rezultate
```

## Dependențe
- OpenCV 4.x
- C++17
- CMake 3.10+

## Note
- Funcțiile recursive pot consuma mult stack pentru regiuni mari
- Pentru imagini mari, preferați `cv::floodFill`
- Threshold-ul controlează cât de similare trebuie să fie pixelii cu seed-ul
- Valori threshold mai mari = regiuni mai extinse
