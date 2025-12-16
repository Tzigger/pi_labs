#!/bin/bash

# Script pentru compilare și rulare Lab 12

echo "================================"
echo "  Laborator 12 - Watershed"
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
if [ -f "Imagini_Laborator/coins.jpg" ]; then
    ./lab12 Imagini_Laborator/coins.jpg
elif [ -f "Imagini_Laborator/weld.jpg" ]; then
    echo "Folosesc weld.jpg..."
    ./lab12 Imagini_Laborator/weld.jpg
else
    echo "Nu s-a găsit nicio imagine. Rulare fără argument..."
    ./lab12
fi
