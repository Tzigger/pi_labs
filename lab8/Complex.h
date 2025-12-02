#ifndef COMPLEX_H
#define COMPLEX_H

#include <cmath>
#include <iostream>

class Complex {
private:
    double real;
    double imag;

public:
    // Constructori
    Complex() : real(0.0), imag(0.0) {}
    Complex(double r, double i = 0.0) : real(r), imag(i) {}
    
    // Getteri
    double getReal() const { return real; }
    double getImag() const { return imag; }
    
    // Setteri
    void setReal(double r) { real = r; }
    void setImag(double i) { imag = i; }
    
    // Modul (magnitudine)
    double magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }
    
    // Fază (argument)
    double phase() const {
        return std::atan2(imag, real);
    }
    
    // Conjugat
    Complex conjugate() const {
        return Complex(real, -imag);
    }
    
    // Operatori aritmetici
    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
    
    Complex operator*(double scalar) const {
        return Complex(real * scalar, imag * scalar);
    }
    
    Complex operator/(double scalar) const {
        if (scalar == 0.0) {
            std::cerr << "Error: Division by zero!" << std::endl;
            return Complex(0, 0);
        }
        return Complex(real / scalar, imag / scalar);
    }
    
    Complex operator/(const Complex& other) const {
        double denominator = other.real * other.real + other.imag * other.imag;
        if (denominator == 0.0) {
            std::cerr << "Error: Division by zero!" << std::endl;
            return Complex(0, 0);
        }
        return Complex(
            (real * other.real + imag * other.imag) / denominator,
            (imag * other.real - real * other.imag) / denominator
        );
    }
    
    // Operatori compuși
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    Complex& operator-=(const Complex& other) {
        real -= other.real;
        imag -= other.imag;
        return *this;
    }
    
    Complex& operator*=(const Complex& other) {
        double temp_real = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        real = temp_real;
        return *this;
    }
    
    Complex& operator*=(double scalar) {
        real *= scalar;
        imag *= scalar;
        return *this;
    }
    
    Complex& operator/=(double scalar) {
        if (scalar == 0.0) {
            std::cerr << "Error: Division by zero!" << std::endl;
            return *this;
        }
        real /= scalar;
        imag /= scalar;
        return *this;
    }
    
    // Operator de atribuire
    Complex& operator=(const Complex& other) {
        if (this != &other) {
            real = other.real;
            imag = other.imag;
        }
        return *this;
    }
    
    // Funcție statică pentru exponențială complexă: e^(i*theta) = cos(theta) + i*sin(theta)
    static Complex fromPolar(double magnitude, double phase) {
        return Complex(magnitude * std::cos(phase), magnitude * std::sin(phase));
    }
    
    // Funcție pentru e^(i*theta)
    static Complex expI(double theta) {
        return Complex(std::cos(theta), std::sin(theta));
    }
    
    // Operator de afișare
    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        os << c.real;
        if (c.imag >= 0) {
            os << " + " << c.imag << "i";
        } else {
            os << " - " << (-c.imag) << "i";
        }
        return os;
    }
};

// Operator de înmulțire scalar * Complex
inline Complex operator*(double scalar, const Complex& c) {
    return c * scalar;
}

#endif // COMPLEX_H