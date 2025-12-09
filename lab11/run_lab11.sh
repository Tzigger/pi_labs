#!/bin/bash

# Script pentru compilare și rulare Lab 11

echo "================================"
echo "  Laborator 11 - Region Growing"
echo "================================"

# Compilare
echo -e "\n[1/2] Compilare..."
cmake . && make

if [ $? -ne 0 ]; then
    echo "Eroare la compilare!"
    exit 1
fi

echo -e "\n[2/2] Rulare..."

# Verificăm dacă există imagini
if [ -f "Imagini_Laborator/weld.jpg" ]; then
    ./lab11 Imagini_Laborator/weld.jpg
elif [ -f "Imagini_Laborator/coins.png" ]; then
    echo "Folosesc coins.png..."
    ./lab11 Imagini_Laborator/coins.png
else
    echo "Nu s-a găsit nicio imagine. Rulare fără argument..."
    ./lab11
fi
