# Classical Mechanics

A comprehensive collection of numerical simulations in Classical Mechanics, implemented in Python.
This module is part of the broader Computational Physics repository and focuses on solving dynamical systems that lack closed-form analytical solutions.

---

## Project Overview
Many physical systems are governed by differential equations that cannot be solved analytically.
This project explores how numerical methods can be used to approximate solutions to such systems with high accuracy.

The simulations emphasize:

* Physical intuition
* Numerical stability and error analysis
* Comparison between analytical and numerical solutions (when available)

All models are implemented using modular, self-contained code, allowing each system to be studied independently.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Kinematics and Dynamics](#kinematics-and-dynamics)
4. [Celestial Mechanics](#celestial-mechanics)
5. [Oscillatory Motion](#oscillatory-motion)
6. [Chaotic Systems and Non-linear Dynamics](#chaotic-systems-and-non-linear-dynamics)
7. [Scattering and Collisions](#scattering-and-collisions)
8. [Advanced Mechanics](#advanced-mechanics)
9. [Numerical Methods](#numerical-methods)

---

## Project Structure

```
Computational Physics/
└── Classical_Mechanics/
    ├── simulation.py
    ├── simulation.ipynb
    └── README.md
```

* `simulation.py`: Main script containing all simulations
* `simulation.ipynb`: Jupyter notebook containing all simulations with the resulting plots by blocks
* Each section is self-contained and executable independently

## Topics Covered

### Kinematics and Dynamics
* **Ideal Projectile Motion:** Simulates drag-free parabolic trajectories using the Runge-Kutta 4 (RK4) method.
* **Realistic Projectile Motion:** Incorporates linear air drag to demonstrate the reduction in both horizontal range and vertical height, modifying the ideal acceleration model. 
* **Cyclist Motion:** Models power-driven motion against quadratic aerodynamic drag. The numerical solution is compared to the exact drag-free solution $v(t)=\sqrt{v_0^2+2Pt/m}$ to highlight the limits imposed by air resistance.
* **Motion in a Rotating Reference Frame:** Visualizes the curved paths created by Coriolis and centrifugal fictitious forces when a straight-line trajectory is viewed from a rotating platform.

### Celestial Mechanics
* **Kepler Orbits:** Simulates planetary motion under an inverse-square gravitational force ($\ddot{\mathbf{r}}=-\frac{GM}{r^3}\mathbf{r}$). The script explores various initial speeds to generate bound elliptical orbits and verifies Hamilton's theorem by plotting the circular velocity hodograph.

### Oscillatory Motion
* **Nonlinear Pendulum:** Compares several integration methods (Euler, Euler-Cromer, Euler-Verlet, and RK4) to analyze numerical error and convergence without relying on the small-angle approximation.
* **Forced Damped Harmonic Oscillator:** Models a mass on a spring subject to both viscous damping and a harmonic driving force: $\ddot{x}=-\omega_0^2x-\gamma\dot{x}+a_0\cos(\omega_d t)$. The simulation explores resonance and visualizes the decay of oscillations in the phase space portrait.
* **Liouville's Theorem:** Demonstrates the conservation of phase-space density for a conservative harmonic oscillator, and contrasts it with the contracting phase space of a dissipative oscillator.

### Chaotic Systems and Non-linear Dynamics
* **Double Pendulum:** A classic example of deterministic chaos featuring two pendula connected in series. Equations of motion are derived from the Lagrangian and solved using `scipy.integrate.odeint` to showcase exponentially diverging trajectories.
* **Van der Pol Oscillator:** Simulates an oscillator with non-linear, velocity-dependent damping: $\ddot{x}-\varepsilon(1-x^2)\dot{x}+x=0$. Highlights the emergence of stable limit cycles in the phase space.
* **Duffing Oscillator:** Models a periodically driven non-linear oscillator within a double-well potential. Demonstrates the system's high sensitivity to driving amplitude ($F$) and its transition into a chaotic regime.
* **Lorenz System:** Solves a simplified version of the Navier-Stokes equations used to model atmospheric convection. Visualizes the canonical 3D "strange attractor" and the butterfly effect characteristic of deterministic chaos.
* **Poincaré Sections:** Employs stroboscopic sampling on a parametrically driven pendulum (Mathieu-type forcing) to reduce the complexity of the phase space. It distinguishes visually between quasiperiodic orbits and the fractal structures of chaotic motion.

### Scattering and Collisions
* **Elastic Collisions in 2D:** Implements a hard-sphere model for two discs interacting only at contact, calculating impulses and verifying analytical deflection angles based on impact parameters.
* **Rutherford Scattering:** Simulates an alpha particle scattering off a heavy nucleus via Coulomb repulsion ($F(r)=k/r^2$), comparing the numerical deflection to the analytical Rutherford formula.
* **Particle Scattering (Morse Potential):** Models particle motion influenced by a Morse potential, showcasing both bound states (capture) and scattering events based on initial conditions.

### Advanced Mechanics
* **Rigid-Body Rotation:** Uses Euler's equations for a torque-free rigid body to demonstrate the "tennis-racket theorem", proving that rotation around the intermediate principal axis is unstable. It also models the stable body-frame precession of a symmetric top.
* **The Brachistochrone Problem:** Numerically calculates and compares transit times to prove that a cycloid is the curve of fastest descent between two points, beating both straight lines and vertical-horizontal drops.

---

## Numerical Methods

To solve these dynamical systems, the `simulation.py` script utilizes the following numerical integrators:
* **Euler Method:** First-order forward integration.
* **Euler-Cromer Method:** Semi-implicit method that conserves a shadow Hamiltonian.
* **Euler-Verlet Method:** Uses finite-difference second derivatives.
* **Runge-Kutta 4th Order (RK4):** A high-accuracy fixed-step integrator used heavily throughout the script.
* **Adaptive Integration:** Leveraging `scipy.integrate.odeint` for complex coupled systems like the double pendulum.
