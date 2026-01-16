#!/bin/bash

# Script pentru compilare și rulare Lab 13

echo "================================"
echo "  Laborator 13 - Modele Culoare"
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
if [ -f "Imagini_Laborator/lena.png" ]; then
    ./lab13 Imagini_Laborator/lena.png
elif [ -f "Imagini_Laborator/fruits.jpg" ]; then
    echo "Folosesc fruits.jpg..."
    ./lab13 Imagini_Laborator/fruits.jpg
elif [ -f "Imagini_Laborator/peppers.png" ]; then
    echo "Folosesc peppers.png..."
    ./lab13 Imagini_Laborator/peppers.png
else
    echo "Nu s-a găsit nicio imagine. Rulare fără argument..."
    ./lab13
fi
