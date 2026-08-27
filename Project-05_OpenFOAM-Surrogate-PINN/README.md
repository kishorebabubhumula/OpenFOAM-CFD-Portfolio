# Project 05: OpenFOAM-to-PyTorch PINN Surrogate Solver
### Scientific Machine Learning (SciML) Fluid Dynamics Accelerators

This repository directory implements a **Data-Driven Physics-Regularized Surrogate Model** that bridges traditional Computational Fluid Dynamics (CFD) with deep neural networks. By training on sparse OpenFOAM data points while enforcing mass conservation laws, the framework creates a sub-millisecond fluid field prediction engine.

---

## 🧠 Integrated Hybrid Architecture

### 1. Unified Dataset & Collocation Fields
* **Data Fields:** 500 spatial points extracted directly from an OpenFOAM 2D Lid-Driven Cavity simulation mapping standard velocity components ($u, v$).
* **Collocation Fields:** 1500 randomized spatiotemporal sample points evaluating continuous domain space.

### 2. Blended Loss Formulation
The optimizer operates across a dual-objective loss constraint to maximize spatial prediction accuracy while ensuring physical validity:
* **Empirical Data Loss Layer:**
  $$\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^N \left[(u_{\text{pred}} - u_{\text{foam}})^2 + (v_{\text{pred}} - v_{\text{foam}})^2\right]$$
* **Incompressible Continuity Loss Residual:**
  $$\mathcal{L}_{\text{physics}} = \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)^2 = 0$$

---

## 📊 Visual Validation Map
The trained model weights reconstruct the entire velocity profile, mapping out high-velocity circulation boundaries, localized stagnation points, and vector field orientations in milliseconds.
