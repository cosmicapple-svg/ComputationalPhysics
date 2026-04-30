
# Computational Physics
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Field](https://img.shields.io/badge/field-computational%20physics-purple)
![GitHub last commit](https://img.shields.io/github/last-commit/cosmicapple-svg/ComputationalPhysics)

A curated collection of Python implementations for key topics in computational physics, including numerical methods, dynamical systems, thermodynamics, classical electrodynamics, and quantum mechanics.

---
*Based on the main project developed for the Computational Physics course at UAQ*

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Jacobi Method for Laplace Equation](#jacobi-method-for-laplace-equation)  
3. [Numerical Implementation: Runge-Kutta Methods](#numerical-implementation-runge-kutta-methods)  
4. [Installation & Usage](#installation--usage)  
5. [References](#references)

---

## Project Overview

This repository was originally developed as part of the **Computational Physics** course taught by Dr. José Alonso López Miranda within the [BSc. in Physics Engineering](https://ingenieria.uaq.mx/index.php/oferta-educativa/licenciaturas/ingenieria-fisica/) at the Autonomous University of Querétaro ([UAQ](https://www.uaq.mx/)).

The project is maintained by:
- [Hugo Suárez](https://linkedin.com/in/hugosuarezrangel)  
- [Bruno Salgado](https://brunosalgado.website/)

Its main goal is to provide clean, reusable, and well-documented implementations of numerical methods commonly used in physics.

---

## Jacobi Method for Laplace Equation

In this approach, the spatial domain is discretized into a 3D grid where each point is labeled by integer indices $(i, j, k)$. The objective is to compute the potential:

$$
V(i,j,k) = V(i\Delta x, j\Delta y, k\Delta z)
$$

Using finite differences, the second derivative in the $x$-direction can be approximated as:

$$
\begin{aligned}
\frac{\partial^2 V}{\partial x^2}
& \approx \frac{V(i+1, j, k) + V(i-1, j, k) - 2V(i,j,k)}{(\Delta x)^2}
\end{aligned}
$$

This symmetric formulation improves numerical stability by considering neighboring points on both sides. The same approximation applies to the $y$ and $z$ directions.

Substituting into Laplace’s equation and solving for $V(i,j,k)$ yields:

$$
\begin{aligned}
V(i,j,k) = \frac{1}{6} [&V(i+1,j,k) + V(i-1,j,k) + V(i,j+1,k) \\
                       &+ V(i,j-1,k) + V(i,j,k+1) + V(i,j,k-1)]
\end{aligned}
$$

Assuming uniform spacing ($\Delta x = \Delta y = \Delta z$), this shows that the value at each point is the average of its neighbors.

### Iterative Solution (Jacobi Method)

To compute the solution:

1. Start with an initial guess $V_0(i,j,k)$  
2. Update all grid points simultaneously using the averaging rule  
3. Repeat iteratively:
   
   $$
   V^{(n+1)}(i,j,k) = \text{average of neighbors of } V^{(n)}
   
   $$
4. Stop when a convergence criterion is satisfied  

This iterative relaxation process is known as the **Jacobi method**, and it is widely used for solving elliptic partial differential equations.

---

## Numerical Implementation: Runge-Kutta Methods

To solve systems of ordinary differential equations (ODEs), this project implements **Runge-Kutta methods**, with a focus on the classical 4th-order scheme (RK4).

A general Runge-Kutta method of order $s$ is defined as:

$$
y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i
$$

$$
k_i = f\left(x_n + c_i h,\; y_n + h \sum_{j=1}^{s} a_{ij} k_j \right)
$$

where $h$ is the step size.

### Common Runge-Kutta Schemes

#### 1. First Order (Euler Method)

$$
y_{n+1} = y_n + h f(x_n, y_n)
$$

#### 2. Second Order (Heun’s Method)

$$
y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)
$$

with:

$$
k_1 = f(x_n, y_n), \quad
k_2 = f(x_n + h, y_n + h k_1)
$$

#### 3. Fourth Order (RK4)

$$
y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

where:

$$
\begin{aligned}
k_1 &= f(x_n, y_n) \\
k_2 &= f(x_n + \tfrac{h}{2}, y_n + \tfrac{h}{2} k_1) \\
k_3 &= f(x_n + \tfrac{h}{2}, y_n + \tfrac{h}{2} k_2) \\
k_4 &= f(x_n + h, y_n + h k_3)
\end{aligned}
$$

Higher-order methods provide greater accuracy at the cost of additional computations. The RK4 method offers an excellent balance between precision and efficiency, making it a standard choice in computational physics.

---

## Installation & Usage

### Python Version

- Python **3.8+** recommended  
- Check your version:
  ```bash
  python --version
  ```

### Dependencies

Required external libraries:

* `numpy`
* `matplotlib`
* `seaborn`

### Setup

1. Clone the repository:

```bash
git clone https://github.com/cosmicapple-svg/ComputationalPhysics.git
cd ComputationalPhysics
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## References

1. Anagnostopoulos, K. N. (2004). *Computational Physics*.
2. Giordano, N. J. (1997). *Computational Physics*.
3. Marion, J. B. (1996). *Classical Dynamics of Particles and Systems*.
4. Sayama, H. (2015). *Introduction to the Modeling and Analysis of Complex Systems*.
5. Chapra, S. C., & Canale, R. P. (2007). *Numerical Methods for Engineers*.
6. Morin, D. (2003). *Classical Mechanics*.
7. Griffiths, D. J. (1995). *Introduction to Quantum Mechanics*.
8. Serway, R. A., & Jewett, J. W. (2009). *Physics for Scientists and Engineers*.
9. Kittel, C. (2004). *Introduction to Solid State Physics*.