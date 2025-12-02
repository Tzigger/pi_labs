# Laborator 9 - Prelucrarea Imaginilor în Domeniul Frecvențelor

## Descriere
Laborator 9 implementează prelucrarea imaginilor în domeniul frecvențelor folosind Transformata Fourier Discretă. 

Aplicația rezolvă următoarele 3 puncte principale:

### 1. Afișarea unei imagini și a transformatei Fourier
- Se folosește funcția `cv::dft` pentru calcularea transformatei Fourier Discrete
- Se afișează imaginea originală alături de spectrul Fourier
- Spectrul este normalizat logaritmic pentru o vizualizare optimă

### 2. Filtrarea în domeniul frecvență pornind de la un filtru gaussian spatial
- Se creează un filtru gaussian spatial 5x5
- Se calculează transformata Fourier a filtrului spatial
- Se înmulțesc valorile TF ale imaginii cu cele ale TF a filtrului
- Se aplică IDFT pentru a obține imaginea filtrată
- Se compară rezultatul cu filtrarea în domeniul spatial

### 3. Filtrarea în domeniul frecvență pornind de la un filtru gaussian în domeniul frecvență
- Se creează direct un filtru gaussian în domeniul frecvenței (formula: H(u,v) = e^(-D²/(2*D0²)))
- Se înmulțesc valorile TF ale imaginii cu filtrul
- Se aplică IDFT pentru a obține imaginea netezită (blurred)

## Structura Proiectului
```
lab9/
├── main_lab9.cpp                    # Program principal cu cele 3 puncte
├── frequency_domain_helpers.h       # Header cu funcții pentru domeniul frecvențelor
├── frequency_domain_helpers.cpp     # Implementarea funcțiilor
├── CMakeLists.txt                   # Configurație CMake
├── run_lab9.sh                      # Script de rulare
├── Imagini_Laborator/               # Imagini de test
│   └── house.pgm
└── output_lab9/                     # Directorul pentru rezultate
```

## Compilare și Rulare

### Metoda 1: Folosind scriptul
```bash
chmod +x run_lab9.sh
./run_lab9.sh
```

### Metoda 2: Manual
```bash
cmake .
make
./lab9
```

### Cu imagine personalizată
```bash
./lab9 cale/catre/imagine.pgm
```

## Funcționalități

### Meniu Interactiv
```
1. Afișează imaginea și transformata Fourier
2. Filtrare în domeniu frecvență pornind de la filtru gaussian spatial
3. Filtrare în domeniul frecvență pornind de la filtru gaussian în domeniul frecvență
0. Ieșire
```

### Punctul 1: Transformata Fourier
- Calculează DFT folosind `cv::dft`
- Afișează spectrul de magnitudine
- Spectrul este shiftat (frecvențe nule în centru)
- Normalizare logaritmică pentru vizualizare

### Punctul 2: Filtrare din domeniul spatial
**Pași:**
1. Creează filtru gaussian spatial 5x5:
   ```
   [1  4  6  4  1]
   [4 16 24 16  4]
   [6 24 36 24  6]
   [4 16 24 16  4]
   [1  4  6  4  1] / 256
   ```
2. Calculează TF a filtrului (redimensionat la dimensiunea imaginii)
3. Înmulțește TF(imagine) * TF(filtru)
4. Aplică IDFT pentru rezultat
5. Compară cu filtrarea în domeniul spatial

**Observație:** Cele două metode (frecvență vs. spatial) ar trebui să producă rezultate similare!

### Punctul 3: Filtrare din domeniul frecvență
**Pași:**
1. Creează filtru gaussian direct în domeniul frecvenței
   - Formula: H(u,v) = e^(-D²/(2*D0²))
   - Parametru D0 configurabil (20-50 recomandat)
2. Înmulțește TF(imagine) * Filtru
3. Aplică IDFT pentru rezultat
4. Afișează imaginea netezită (blurred)

**Parametru D0:**
- Valori mici (10-20): Netezire puternică
- Valori medii (30-50): Netezire moderată
- Valori mari (70-100): Netezire slabă

## Funcții OpenCV Utilizate
- `cv::dft` - Transformata Fourier Discretă directă
- `cv::idft` - Transformata Fourier Inversă
- `cv::magnitude` - Calcularea magnitudinii complexe
- `cv::log` - Scalare logaritmică
- `cv::normalize` - Normalizare pentru afișare
- `cv::split` / `cv::merge` - Manipularea canalelor complexe
- `cv::multiply` - Multiplicare element cu element
- `cv::filter2D` - Filtrare în domeniul spatial

## Cerințe
- OpenCV 4.x
- CMake 3.10+
- C++17

## Teoria Transformatei Fourier
Transformata Fourier permite analiza și modificarea imaginilor în domeniul frecvențelor:
- **Frecvențe joase**: Variații lente de intensitate (zone omogene)
- **Frecvențe înalte**: Variații rapide (muchii, detalii fine)

Formula DFT 2D:
```
F(u,v) = Σ Σ f(x,y) * e^(-j*2π*(ux/M + vy/N))
```

Formula IDFT 2D:
```
f(x,y) = (1/MN) * Σ Σ F(u,v) * e^(j*2π*(ux/M + vy/N))
```

Formula filtrului gaussian în domeniul frecvenței:
```
H(u,v) = e^(-D²/(2*D0²))
unde D = sqrt((u-M/2)² + (v-N/2)²)
```

## Avantaje Filtrare în Domeniul Frecvenței
1. **Eficiență**: Pentru imagini mari, FFT este mai rapidă decât convoluția directă
2. **Control direct**: Manipulare precisă a frecvențelor
3. **Înțelegere**: Vizualizarea conținutului frecvențial al imaginii
4. **Simplitate**: Înmulțire simplă în loc de convoluție

## Comparație: Domeniul Spatial vs. Frecvență
- **Domeniul Spatial**: Convoluție directă cu kernel-ul
- **Domeniul Frecvenței**: 
  1. DFT(imagine)
  2. DFT(filtru) sau filtru direct în frecvență
  3. Multiplicare
  4. IDFT(rezultat)

Teoretic, ambele metode ar trebui să dea rezultate identice (cu mici diferențe numerice)!
