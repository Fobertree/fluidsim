Targeting implementations of:
- Eulerian
  - Fluid
  - Flame
  - Smoke
  - Air
- FLIP (Fluid Implicit Particle)
- APIC (Affine Particle-In-Cell)

Since we just want pressure-based, draw a full-screen quad for teh vertex then let fragment shader map the intensity

push constants

https://docs.vulkan.org/tutorial/latest/08_Loading_models.html
- Need to write collision/physics wrapper for tinyobj or gltf

SDF/Level set collision from triangle mesh

Rendering
- Rasterization
- Particle splatting
- Volume Rendering
- Marching cubes
- https://www.youtube.com/watch?v=kOkfC5fLfgE

Tentatively Marching Cubes

4.3.3: Finding Signed Distance for a Triangle Mesh

Need to update MAC from velocity field (37)

## Vulkan Side
Raymarching impl

- Object Management
- Drag & Drop