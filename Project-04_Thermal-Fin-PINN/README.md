# Project 04: Convective Heat Transfer Pin-Fin Simulation
### Scientific Machine Learning (SciML) Heat Sink Analytics

This repository directory implements a **Physics-Informed Neural Network (PINN)** to simulate steady-state 1D/2D thermal profiles through extended convective cooling surfaces (pin-fins) without requiring an underlying mesh grid.

---

## 🧠 Framework Governing Physics

### 1. Neural Network Domain
* **Inputs (2):** Spatial dimension along the fin axis ($x$) and ambient convective temperature profile ($T_{\infty}$).
* **Outputs (1):** Local temperature variable ($T$) across the solid body.

### 2. Embedded Thermal Loss Residuals
The optimizer uses automatic differentiation to strictly enforce Fourier's Law of heat conduction balanced against Newton's Law of Cooling:
* **Governing Differential Equation (1D Fin Approximation):**
  $$\mathcal{L}_{\text{physics}} = \frac{d^2 T}{dx^2} - \frac{h P}{k A_c} (T - T_{\infty}) = 0$$
  * *Where:* $h$ = convective coefficient, $P$ = fin perimeter, $k$ = thermal conductivity, $A_c$ = cross-sectional area.

### 3. Structural Boundary Targets
* **Base Condition (Dirichlet):** Hardcoded temperature constraints at the root matching the core engine casing ($T(0) = T_{\text{base}}$).
* **Tip Condition (Neumann):** Insulated adiabatic fin tip constraint enforcing zero gradient ($\frac{dT}{dx} = 0$ at $x = L$).

