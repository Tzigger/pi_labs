#!/bin/bash

# Script pentru rularea laboratorului 7

# Compilare
echo "Compilare laborator 7..."
cmake .
make

# Verificare compilare
if [ $? -eq 0 ]; then
    echo "Compilare reusita!"
    echo ""
    echo "Rulare laborator 7..."
    echo ""
    
    # Rulare cu imaginea default sau cu parametrul primit
    if [ -n "$1" ]; then
        ./lab7 "$1"
    else
        ./lab7
    fi
else
    echo "Eroare la compilare!"
    exit 1
fi
