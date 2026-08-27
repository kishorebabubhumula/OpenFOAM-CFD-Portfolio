# Project 03: Physics-Informed Neural Network (PINN) Fluid Simulator
### Himalayan Flash Flood Resilience Analytics Model

This project implements a modern **Scientific Machine Learning (SciML)** framework to simulate high-energy fluid propagation (such as a Glacial Lake Outburst Flood or mountain avalanche surge) using a **Physics-Informed Neural Network (PINN)**.

---

## 🧠 Framework Architecture

### 1. Neural Network Structure
* **Inputs (3):** Normalized spatial coordinates and time $(x, y, t) \in \mathbb{R}^3$.
* **Outputs (3):** Fluid column thickness $h$, velocity along the X-axis $u$, and velocity along the Y-axis $v$.
* **Hidden Layers:** 2 dense layers with 64 neurons each, utilizing smooth `Tanh` activations to compute continuous partial derivatives via automatic differentiation.

### 2. Multi-Objective Physics Loss Functions
The optimizer embeds exact partial differential equations directly into the network training process:
* **Mass Conservation (Continuity Equation):**
  $$\mathcal{L}_{\text{mass}} = \frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = 0$$
* **Momentum Conservation (Navier-Stokes + Gravity Incline Driving Forces):**
  $$\mathcal{L}_{\text{mom\_x}} = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + g \frac{\partial h}{\partial x} - g \cdot \text{slope}_x = 0$$
  $$\mathcal{L}_{\text{mom\_y}} = \frac{\partial v}{\partial t} + v \frac{\partial v}{\partial y} + g \frac{\partial h}{\partial y} - g \cdot \text{slope}_y = 0$$

### 3. Integrated Features
* **Interactive Timeline GUI:** Built with `ipywidgets` to dynamically query trained neural network model weights.
* **Flow Field Tracking Vectors:** Rendered with normalized Matplotlib `quiver` indicators to monitor directional debris velocity.
* **Infrastructure Assessment:** Integrates a boundary masking layer to evaluate how surge dynamics interact with physical block obstacles like downstream villages and bridges.
