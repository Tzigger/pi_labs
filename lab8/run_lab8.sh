#!/bin/bash

# Script pentru rularea laboratorului 8

# Compilare
echo "Compilare laborator 8..."
cmake .
make

# Verificare compilare
if [ $? -eq 0 ]; then
    echo "Compilare reusita!"
    echo ""
    echo "Rulare laborator 8..."
    echo ""
    
    # Rulare cu imaginea default sau cu parametrul primit
    if [ -n "$1" ]; then
        ./lab8 "$1"
    else
        ./lab8
    fi
else
    echo "Eroare la compilare!"
    exit 1
fi
