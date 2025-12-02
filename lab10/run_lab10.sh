#!/bin/bash

# Script pentru compilare și rulare Lab 10

echo "=== Compilare Lab 10 - Segmentarea Imaginilor ==="

# Compilăm
cmake . -B build 2>/dev/null || cmake .
make -j4 2>/dev/null || make

# Verificăm dacă compilarea a reușit
if [ -f "./lab10" ]; then
    echo ""
    echo "✓ Compilare reușită!"
    echo ""
    
    # Rulăm cu imaginea specificată sau implicit
    if [ -n "$1" ]; then
        ./lab10 "$1"
    else
        # Încercăm să găsim o imagine
        if [ -f "Imagini_Laborator/cameraman.png" ]; then
            ./lab10 Imagini_Laborator/cameraman.png
        elif [ -f "../lab9/Imagini_Laborator/cameraman.png" ]; then
            ./lab10 ../lab9/Imagini_Laborator/cameraman.png
        else
            echo "Specificați calea către imagine:"
            echo "./run_lab10.sh <cale_imagine>"
            ./lab10
        fi
    fi
else
    echo "✗ Eroare la compilare!"
    exit 1
fi
