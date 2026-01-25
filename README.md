# NURBS-Splatting : Differentiable 3D Scene Reconstruction

##  Overview

This project focuses on **Differentiable Scene Reconstruction** using **NURBS (Non-Uniform Rational B-Splines)**. While most modern reconstruction methods (like NeRF or Gaussian Splatting) rely on discrete or volumetric representations, this engine explores the potential of parametric surfaces to define 3D geometry.

### Why NURBS?

* **Compactness:** Represents complex, smooth surfaces with a fraction of the parameters required by dense meshes.
* **Mathematical Exactness:** Capable of representing conic sections (circles, spheres) perfectly, unlike linear approximations.
* **Native Differentiability:** Allows for direct gradient-based optimization of control points and weights through a differentiable rendering pipeline.

### Current Status

This repository is an **active work-in-progress**. The current implementation features a custom, fully differentiable **Cox-de Boor kernel** and is currently being integrated with the **NVIDIA Kaolin** library for high-performance rendering.

## Core Strategy: The 4-Step Optimization Loop

NURBS optimization is inherently non-linear and prone to stagnation in local minima. To address this, the project implements a specialized feedback loop, referred to as the **"Brain Graft" strategy**, which dynamically adapts the surface's complexity based on its learning performance.

### 1. Train

The engine performs gradient-based optimization on control point positions and weights. By utilizing a **differentiable rendering pipeline** (via NVIDIA Kaolin), the model learns the 3D geometry directly from 2D observations or 3D target losses.

### 2. Inspect

Rather than training blindly, the system monitors convergence. It analyzes the **loss trajectory** and gradient variance to detect "stagnation"—moments where the model stops improving despite a high remaining error.

### 3. Filter

Not all parts of a surface require the same amount of detail. The system calculates a **local error map** to isolate specific NURBS patches or regions where the reconstruction is inaccurate. This prevents unnecessary global computations.

### 4. Act

When stagnation is detected in high-error regions, the system triggers a refinement:

* **Targeted Upsampling:** Strategic knot insertion or degree elevation to increase local degrees of freedom.
* **Optimizer Reset:** A local or global "Brain Graft" where the optimizer state is reset to prevent momentum from hindering the new, refined geometry.

