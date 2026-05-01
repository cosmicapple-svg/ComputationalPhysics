# Quantum Mechanics

A comprehensive collection of numerical simulations in Quantum Mechanics, implemented in Python.
This module is part of the broader Computational Physics repository and focuses on solving the Schrödinger equation, simulating time evolution, applying approximation methods, and exploring fundamental quantum phenomena.

---

## Project Overview
Quantum Mechanics governs the behaviour of matter and light at the atomic and subatomic scales. 
This project explores how numerical methods can be used to simulate quantum systems, solve for bound states and scattering amplitudes, and visualize complex phenomena like wave-packet dynamics, quantum tunneling, and spin precession.

The simulations emphasize:

* Physical intuition for wavefunctions, probability densities, and energy quantization.
* Numerical stability and unitarity for time-dependent evolution.
* Comparison between exact analytical solutions and numerical approximations.
* Connections between fundamental quantum models and solid-state physics (band structure, tight-binding).

All models are implemented using modular, self-contained code, allowing each system to be studied independently.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Bound States & Potential Wells](#bound-states--potential-wells)
4. [Fundamental Quantum Systems](#fundamental-quantum-systems)
5. [Scattering & Tunneling](#scattering--tunneling)
6. [Time Evolution & Dynamics](#time-evolution--dynamics)
7. [Approximation Methods](#approximation-methods)
8. [Solid State & Molecular Models](#solid-state--molecular-models)
9. [Advanced Topics](#advanced-topics)
10. [Numerical Methods](#numerical-methods)

---

## Project Structure

```
Computational Physics/
└── Quantum Mechanics/
    ├── simulation.py
    ├── simulation.ipynb
    └── README.md
```

* `simulation.py`: Main script containing all simulations
* `simulation.ipynb`: Jupyter notebook containing all simulations with the resulting plots by blocks
* Each section is self-contained and executable independently

---

## Topics Covered

### Bound States & Potential Wells

* **Particle in a 1-D and 2-D Infinite Square Well:** Visualises exact normalised eigenfunctions, probability densities, and the energy ladder ($E_n \propto n^2$). The 2-D simulation highlights contour mappings and energy level degeneracies for a square box.
* **Finite Square Well:** Solves the transcendental equations $\kappa = k\tan(ka)$ (even) and $\kappa = -k\cot(ka)$ (odd) graphically and numerically to find bound-state energies and constructs the decaying exterior and oscillating interior wavefunctions.
* **Dirac Delta Potential:** Analyzes single ($V(x) = -\alpha\delta(x)$) and double delta-function wells. Computes the exact bound state for a single well, and solves the transcendental equations for the symmetric (bonding) and antisymmetric (antibonding) energy splitting in a double well.
* **Infinite Spherical Well:** Calculates energy levels by finding the roots of spherical Bessel functions $j_l(k_{nl}R) = 0$ and plots the corresponding radial wavefunctions.

### Fundamental Quantum Systems

* **Quantum Harmonic Oscillator:** Constructs the equally spaced energy spectrum $E_n = \hbar\omega(n + 1/2)$ and exact wavefunctions using Hermite polynomials. Verifies classical turning points and the Heisenberg uncertainty principle.
* **Hydrogen Atom:** Evaluates and plots the exact radial wavefunctions $R_{nl}(r)$ and radial probability densities $P(r) = r^2|R_{nl}|^2$ using associated Laguerre polynomials, marking the most probable radii (e.g., $a_0$ for the 1s state).
* **Rigid Rotor:** Visualises the probability densities of the spherical harmonics $|Y_l^m(\theta,\phi)|^2$ on 3-D polar plots and displays the $(2l+1)$-degenerate rotational energy spectrum $E_l \propto l(l+1)$.
* **Spin-½ Precession & Rabi Oscillations:** Simulates a spin-½ particle in a magnetic field with a resonant transverse drive. Integrates the effective Hamiltonian in the rotating frame to plot the state vector trajectory on the **Bloch Sphere** and demonstrates Rabi oscillations and lineshape vs. detuning.

### Scattering & Tunneling

* **Quantum Tunneling through a Rectangular Barrier:** Computes the exact transmission coefficient $T$ for a single barrier and generalises to multiple barriers (quantum-well heterostructures) using the **Transfer Matrix method**.
* **WKB Approximation:** Estimates the tunneling transmission coefficient $T_{\rm WKB} \approx \exp(-2\int \kappa(x)dx)$ for arbitrary barrier shapes, comparing the rectangular barrier result with the exact transfer-matrix solution, and testing triangular (Fowler-Nordheim) and parabolic barriers.

### Time Evolution & Dynamics

* **Gaussian Wave-Packet:** Evolves a free-particle Gaussian wave-packet in time using the **split-step Fast Fourier Transform (FFT)** method. Demonstrates spatial spreading $\sigma(t)$ and group velocity while maintaining a constant momentum-space distribution.
* **Time-Dependent Schrödinger Equation (TDSE):** Implements the unconditionally stable **Crank–Nicolson FDTD** scheme to simulate a wave-packet scattering off a rectangular barrier, verifying unitarity (norm conservation) by tracking the transmitted and reflected probability currents.
* **Coherent States:** Visualises the minimum-uncertainty Gaussian state of the harmonic oscillator, showing how its centre oscillates according to classical mechanics while its spatial width remains strictly constant.

### Approximation Methods

* **Variational Method:** Uses trial functions to compute upper bounds for the ground-state energies of the Infinite Square Well and Quantum Harmonic Oscillator, optimising variational parameters analytically and numerically.
* **Perturbation Theory:** Applies Rayleigh-Schrödinger perturbation theory to calculate 1st and 2nd order energy corrections for the Infinite Square Well subjected to linear ramp and Gaussian bump perturbations.

### Solid State & Molecular Models

* **Lennard-Jones Potential:** Finds the bound states of an Argon dimer potential well $V(r) = 4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$ using the **Numerov shooting method** and plots the resulting vibrational levels.
* **Kronig–Penney Model:** Solves the periodic barrier transcendental equation to derive the electronic band structure $E(k)$ and band gaps for a 1-D periodic crystal, illustrating Bloch's theorem.
* **Tight-Binding Model:** Computes the exact dispersion relation for a 1-D infinite chain, compares open (OBC) and periodic (PBC) boundary conditions for finite chains, and explores Anderson localisation by calculating the Inverse Participation Ratio (IPR) under on-site disorder.

### Advanced Topics

* **Path Integral Monte Carlo (PIMC):** Simulates the Feynman path integral for a free particle in imaginary time using Metropolis-Hastings sampling of ring polymers, recovering the exact Gaussian thermal centroid distribution and classical thermal de Broglie variance.

---

## Numerical Methods

To solve these quantum systems, the `simulation.py` script utilises the following numerical methods:

* **Direct Analytic Solutions:** Scipy special functions (`scipy.special`) are extensively used for Hermite polynomials, Laguerre polynomials, Spherical Harmonics, and Spherical Bessel functions.
* **Brent's Method (`scipy.optimize.brentq`):** Used for finding the roots of transcendental equations (Finite Square Well, Dirac Delta, Spherical Well zeros) and for the Numerov shooting method matching.
* **Numerov Integration:** Second-order differential equation solver, exact for Schrödinger-type equations, used to find bound states in the Lennard-Jones potential.
* **Transfer-Matrix Method:** Matrix multiplication techniques for solving piecewise-constant potentials and multi-barrier tunneling.
* **Fast Fourier Transform (FFT):** Employed for the split-step exact time evolution of free-particle wave-packets.
* **Gaussian Quadrature (`scipy.integrate.quad`):** High-precision integration for variational energy bounds and perturbation theory matrix elements.
* **Runge–Kutta 4th Order (RK4):** Integrates the Bloch equations for spin-½ precession and Rabi oscillations.
* **Crank–Nicolson FDTD:** An implicit, unitary, and unconditionally stable finite-difference time-domain solver used for the Time-Dependent Schrödinger Equation (`scipy.linalg.solve_banded` for tridiagonal matrices).
* **Metropolis–Hastings Algorithm:** Used in the Path Integral Monte Carlo simulation to sample imaginary-time path configurations.
* **Matrix Diagonalisation (`numpy.linalg.eigh`):** Used to compute the eigenvalues and eigenstates for Tight-Binding Hamiltonian matrices.