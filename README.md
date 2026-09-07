# NURBS-Splatting

Differentiable 3D scene reconstruction that replaces the discrete Gaussians of 3D Gaussian Splatting with continuous parametric NURBS surfaces, and Spherical Harmonics with a Cook-Torrance BRDF driven by explicit light sources.

## Motivation

3D Gaussian Splatting produces excellent renderings, but its output is hard to use downstream. Three limitations motivate this work:

- **Baked lighting.** Spherical Harmonics encode the radiance observed at capture time. Relighting a reconstructed scene requires re-optimizing it entirely.
- **Geometric inefficiency.** Gaussians have smooth, unbounded falloff, so representing sharp edges takes an ever-growing number of primitives.
- **No explicit geometry.** Gaussians define no closed surface or coherent topology — nothing that can be exported to Blender, Maya, or a CAD tool.

NURBS are the industry standard for parametric surfaces precisely because they avoid all three: they define explicit topology, represent conic sections exactly rather than by approximation, and describe complex curved surfaces compactly. This project asks whether they can be optimized directly from multi-view images the way Gaussians are.

## Method

**Differentiable NURBS surfaces.** Control points, rational weights, and knot vectors are all learnable parameters, alongside per-surface material properties (albedo, roughness, metalness) — geometry and material are recovered by the same gradient descent. Basis functions implement the Cox-de Boor recursion in tensor operations, with dedicated handling of the 0/0 case the rational form produces. Surface normals are derived analytically from the partial derivatives rather than estimated from a mesh, and tessellation resolution adapts to the surface's second derivative.

**Physically-based shading.** A full Cook-Torrance BRDF written from scratch: GGX normal distribution, Fresnel-Schlick approximation, Smith geometry term, with diffuse and specular contributions separated through the Fresnel term. Rendering happens in HDR and is compressed to display space through a differentiable ACES filmic tone mapping curve — a hard clip would destroy the specular gradients the optimizer needs early on. For the same reason, dot products use a softplus-based soft clamp rather than a hard `clamp`, which would zero out gradients wherever a surface faces away from the light.

**Annealed Gaussian Smoothing.** Optimizing geometry and PBR properties jointly produces a chaotic loss landscape: specular highlights are high-frequency functions of both, so a slightly wrong geometry can yield a gradient pointing away from the solution entirely. The fix is to blur both rendered and target images before computing the loss, with a Gaussian kernel whose width decays across training (σ from 3.0 to 0.1). Early on, only low-frequency structure survives the blur — global placement, overall light distribution — so the optimizer settles topology first; as σ decays, BRDF detail is progressively reintroduced. A Huber loss on top limits the influence of poorly-fitted specular pixels on the global gradient.

This is the component the method depends on, not a refinement: the ablation in `experiences/` shows that without it, Huber loss alone fails to converge geometrically — surfaces degrade over iterations and only coarse colour is recovered.

## Results

Validated on synthetic scenes under controlled initialization: target images are rendered from a known NURBS scene, then the same patches are perturbed with Gaussian noise and the optimizer must recover the original configuration from the multi-view photometric signal alone. This isolates the optimization core from the unsolved problem of automatic primitive placement.

On a three-sphere scene supervised by four views over 2500 AdamW iterations, all views converge simultaneously, low-frequency structure first, specular detail last — matching the expected annealing dynamics. Gradient norms decay monotonically across every parameter group with no catastrophic spikes, confirming the smoothing schedule regularizes the landscape as intended.

`experiences/` contains the recorded runs: convergence GIFs, gradient flow traces, and the ablation frames.

## Structure

| Path | Role |
|---|---|
| `src/geometry/nurbs.py` | Differentiable NURBS surface, Cox-de Boor basis, adaptive tessellation |
| `src/rendering/point_lights.py` | Cook-Torrance BRDF, point light model |
| `src/rendering/scene.py` | Scene aggregation, view transforms, rasterization, ACES tone mapping |
| `experiences/kaolin_nurbs.ipynb` | Training loop, annealed smoothing schedule, gradient tracking |
| `experiences/` | 2D NURBS exploration, primitive fitting, recorded results |
| `notes/architecture.md` | Design notes |

## Tech stack

PyTorch, NVIDIA Kaolin, nvdiffrast, jaxtyping + beartype (runtime tensor shape checking).

## Status and limitations

Proof of concept from a one-year academic research track. The core hypothesis holds — NURBS surfaces can be optimized from a multi-view differentiable photometric signal, and the Cook-Torrance integration correctly separates geometry from lighting, which is what makes relighting possible without re-optimization. What remains open:

- **Initialization is manual.** Patches are derived from the target scene. Determining their number, position, and orientation from images alone is the prerequisite for real scenes.
- **Topology is fixed.** No mechanism subdivides poorly-fitted patches or merges redundant ones during optimization.
- **Validation is synthetic.** Geometrically simple scenes only; no evaluation on standard benchmarks, so no quantitative comparison against the state of the art yet.

The main planned direction is an MCMC mechanism governing patch creation, deletion, and movement — which would replace manual initialization and add the missing topological adaptation in one step. Beyond that: replacing rasterization with differentiable ray tracing, which the Cook-Torrance model would need to reach shadows and inter-object reflections.

## Report

A full write-up of the method, experiments, and related work is available in the accompanying research track report.