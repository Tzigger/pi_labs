#!/bin/bash

# Script pentru rularea laboratorului 9 - Prelucrarea imaginilor in domeniul frecventelor

# Compilare
echo "Compilare laborator 9..."
cmake .
make lab9

# Verificare compilare
if [ $? -eq 0 ]; then
    echo "Compilare reusita!"
    echo ""
    echo "Rulare laborator 9..."
    echo ""
    
    # Rulare cu imaginea default sau cu parametrul primit
    if [ -n "$1" ]; then
        ./lab9 "$1"
    else
        ./lab9
    fi
else
    echo "Eroare la compilare!"
    exit 1
fi
