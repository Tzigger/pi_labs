#!/bin/bash

# Script pentru compilare și rulare Laborator 7 (versiunea cu GUI)
# Prelucrarea Imaginilor în Domeniul Spațial

set -e  # Exit on error

echo "========================================"
echo "Laborator 7 - Prelucrarea Imaginilor"
echo "Versiunea cu GUI (OpenCV imshow)"
echo "========================================"
echo ""

# Verificare dacă OpenCV este instalat
echo "1. Verificare OpenCV..."
if ! pkg-config --exists opencv4; then
    echo "✗ OpenCV nu este instalat!"
    echo "Instalează OpenCV cu: brew install opencv"
    exit 1
fi
echo "✓ OpenCV versiunea: $(pkg-config --modversion opencv4)"
echo ""

# Creare director build
echo "2. Creare director build..."
mkdir -p build
echo "✓ Director creat"
echo ""

# Compilare
echo "3. Compilare program..."
OPENCV_FLAGS=$(pkg-config --cflags opencv4)
OPENCV_LIBS=$(pkg-config --libs opencv4)

clang++ -std=c++17 -Wall $OPENCV_FLAGS main.cpp $OPENCV_LIBS -o build/negate_coins

if [ $? -eq 0 ]; then
    echo "✓ Compilare reușită"
else
    echo "✗ Eroare la compilare!"
    exit 1
fi
echo ""

# Rulare
echo "4. Rulare program cu GUI..."
echo ""
echo "INSTRUCȚIUNI:"
echo "  - Alege opțiunea din meniu (1-6)"
echo "  - Apasă orice tastă în ferestrele cu imagini pentru a continua"
echo "  - Alege 0 pentru a ieși din program"
echo ""
echo "========================================"
echo ""

./build/save_to_files

echo ""
echo "========================================"
echo "✓ Program încheiat"
echo ""
