# Thermodynamics

A comprehensive collection of numerical simulations in Thermodynamics, implemented in Python.
This module is part of the broader Computational Physics repository and focuses on solving equations of state, transport phenomena, and statistical physics problems.

---

## Project Overview
Thermodynamics governs the behaviour of energy, temperature, and matter across scales ranging from individual molecules to macroscopic engineering systems.
This project explores how numerical methods can be used to simulate thermodynamic processes, solve partial differential equations for heat transport, and recover statistical behaviour from first principles.

The simulations emphasize:

* Physical intuition across classical, statistical, and continuum thermodynamics
* Numerical stability and convergence for parabolic PDEs
* Comparison between analytical and numerical solutions (when available)
* Connections between microscopic (kinetic theory) and macroscopic (continuum) descriptions

All models are implemented using modular, self-contained code, allowing each system to be studied independently.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Equations of State](#equations-of-state)
4. [Thermodynamic Cycles and Processes](#thermodynamic-cycles-and-processes)
5. [Statistical Mechanics](#statistical-mechanics)
6. [Heat Conduction and Diffusion](#heat-conduction-and-diffusion)
7. [Radiative and Convective Cooling](#radiative-and-convective-cooling)
8. [Stochastic Thermodynamics](#stochastic-thermodynamics)
9. [Numerical Methods](#numerical-methods)

---

## Project Structure

```
Computational Physics/
└── Thermodynamics/
    ├── simulation.py
    ├── simulation.ipynb
    └── README.md
```

* `simulation.py`: Main script containing all simulations
* `simulation.ipynb`: Jupyter notebook containing all simulations with the resulting plots by blocks
* Each section is self-contained and executable independently

---

## Topics Covered

### Equations of State

* **PVT Diagram — Ideal Gas:** Plots isotherms $P = nRT/V$ across a range of temperatures and renders the full three-dimensional PVT surface, illustrating the hyperbolic structure of ideal-gas isotherms and the linear pressure–temperature relationship at fixed volume.
* **Van der Waals Gas & Maxwell Construction:** Extends the ideal-gas model with molecular volume ($b$) and intermolecular attraction ($a$) corrections: $\left(P + a/V_m^2\right)(V_m - b) = RT$. Below the critical temperature the isotherm develops an unphysical oscillation (spinodal region); the **Maxwell equal-area construction** replaces it with the correct phase-equilibrium tie-line, recovering the liquid–vapour coexistence curve. Parameters are those of CO₂ ($T_c = 304.2$ K, $P_c = 7.39$ MPa).

### Thermodynamic Cycles and Processes

* **Carnot Cycle:** Constructs the four reversible strokes of the most efficient heat engine operating between reservoirs at $T_H$ and $T_C$: isothermal expansion, adiabatic expansion, isothermal compression, and adiabatic compression. The enclosed area on the P-V diagram equals the net work output, and the efficiency is verified against the theoretical limit $\eta = 1 - T_C / T_H$.
* **Adiabatic vs. Isothermal Expansion:** Compares the P-V paths and work output for the same volume change under isothermal ($PV = \text{const}$) and adiabatic ($PV^\gamma = \text{const}$) conditions. The steeper adiabat reflects the temperature drop experienced when no heat is exchanged, resulting in less work for any expansion ratio. Three values of $\gamma$ (monatomic, diatomic, triatomic) are superimposed to show the dependence on degrees of freedom.

### Statistical Mechanics

* **Maxwell–Boltzmann Speed Distribution:** Plots the probability density $f(v) \propto v^2 \exp(-mv^2/2k_BT)$ for several gases (H₂, He, N₂, O₂, CO₂) at a fixed temperature and for N₂ across a wide temperature range. The three characteristic speeds — most probable $v_p$, mean $\langle v \rangle$, and root-mean-square $v_\text{rms}$ — are annotated on each curve.

### Heat Conduction and Diffusion

* **1D Heat Conduction in a Thin Bar:** Discretises the parabolic PDE $\partial_t T = \alpha\, \partial_{xx} T$ using the Forward-Time Centred-Space (FTCS) scheme and evolves two physically distinct scenarios: (A) Dirichlet boundary conditions with an initially uniform profile, where the solution relaxes toward the linear steady state; and (B) Neumann (insulated) boundary conditions with a Gaussian hot spot, where the energy spreads symmetrically until thermal equilibrium is reached. The stability parameter $r = \alpha\,\Delta t / (\Delta x)^2 \leq 0.5$ is enforced throughout.
* **2D Heat Diffusion on a Rectangular Grid:** Extends the FTCS method to a two-dimensional plate ($60\times60$ grid) with a hot strip along the bottom edge, a cold top boundary, and insulated lateral walls. Snapshots at logarithmically spaced times reveal the progressive penetration of the thermal front from the hot boundary and the approach to the two-dimensional steady state.
* **2D Diffusion on a Circular Drum:** Solves the heat equation on a disk of radius $R$ with homogeneous Dirichlet boundary conditions using a **spectral Bessel-mode decomposition**: $T(r,\theta,t) = \sum_{m,n} J_m(\lambda_{mn} r)(A_{mn}\cos m\theta + B_{mn}\sin m\theta)\, e^{-\alpha\lambda_{mn}^2 t}$. The initial Gaussian hot spot is projected onto the eigenbasis by numerical quadrature, and the solution is evaluated on a Cartesian grid at successive times to visualise the circular symmetry of the decay.

### Radiative and Convective Cooling

* **Newton's Law of Cooling vs. Stefan–Boltzmann Radiation:** Compares three cooling models for a small steel sphere quenched from 1200°C: pure convective cooling ($\dot{T} \propto T - T_\infty$, exponential decay), pure radiative cooling ($\dot{T} \propto T^4 - T_\infty^4$, dominant at high temperatures), and a physically realistic combined model. All three ODEs are integrated with RK4. The Newton case is cross-validated against its analytical exponential solution $T(t) = T_\infty + (T_0 - T_\infty)e^{-t/\tau}$.

### Stochastic Thermodynamics

* **Brownian Motion and the Diffusion Coefficient:** Simulates $N = 1000$ independent particles in 2-D undergoing Gaussian-distributed random displacements calibrated to the Stokes–Einstein diffusion coefficient $D = k_BT / (6\pi\eta a)$. The mean squared displacement $\langle r^2(t)\rangle = 4Dt$ is computed ensemble-averaged and verified against theory on a log–log scale. The final displacement histogram is compared to the analytical Rayleigh distribution, closing the loop between the stochastic microscopic description and the macroscopic diffusion equation.

---

## Numerical Methods

To solve these thermodynamic systems, the `simulation.py` script utilises the following numerical methods:

* **Runge–Kutta 4th Order (RK4):** High-accuracy fixed-step ODE integrator used for the Carnot cycle, adiabatic expansion, and all cooling ODEs.
* **Forward-Time Centred-Space (FTCS):** Explicit finite-difference scheme for parabolic PDEs; applied to the 1D bar, 2D rectangular grid, and (implicitly) the spectral drum problem.
* **Spectral Bessel-Mode Decomposition:** Projection of the initial condition onto the exact eigenbasis of the Laplacian on a disk; yields exponentially accurate time evolution without any time-stepping error on the spatial modes.
* **Maxwell Equal-Area Construction:** Brent root-finding combined with adaptive quadrature (`scipy.integrate.quad`) to locate the phase-equilibrium pressure on van der Waals isotherms below the critical temperature.
* **Monte-Carlo Random Walk:** Ensemble simulation of Gaussian-step diffusion; used to verify the Einstein relation and the Rayleigh displacement distribution from first principles.