# Overview

## CardiacTissue

The `CardiacTissue` class is used to represent myocardial tissue and its
structural properties in simulations. It includes several key attributes that
define the characteristics and behavior of the cardiac mesh used in
finite-difference calculations.

### Mesh

The mesh attribute of the `CardiacTissue` class is a 2D or 3D array that
represents the structure of the cardiac tissue. Each element of the mesh
represents a cell in the tissue, with different values assigned to different
types of tissue:

- Cardiomyocytes: Represented by a value of 1.
- Fibrotic tissue: Represented by a value of 2.
- Empty nodes (boundary): Represented by a value of 0.

The mesh can be initialized with a specific size and shape using the
`CardiacTissue2D` or `CardiacTissue3D` classes. The mesh can be modified
by setting the values of individual cells or by using the `add_boundaries()`
method to add a border of empty nodes around the tissue.

### Conductivity

The `conductivity` attribute of the `CardiacTissue` class is an array that
assigns different conductivities to different regions of the mesh. This allows
users to simulate the effects of fibrosis on wave propagation in the heart by
modifying the conductivity values in the tissue.

### Fibrosis

Fibrosis is a common feature of cardiac tissue that can be modeled in
simulations using Finitewave.

- Fibrotic tissue is represented by a conductivity array, which assigns 
  different conductivities to different regions of the mesh. This allows users
  to simulate the effects of fibrosis on wave propagation in the heart.

- Another way to model fibrosis is to set `mesh` values to 2 (empty nodes).

### Fibers

By default, Finitewave uses an isotropic conduction model, where the
conductivity is the same in all directions. And the fibers are not used in the
simulation. The default stencil is `IsotropicStencil2D` or
`IsotropicStencil3D`, which is used to calculate the weights for
the divergence kernels in the finite-difference scheme.

To simulate anisotropic conduction, users can define the orientation of
cardiac fibers in the tissue. This is done by setting the `fibers` attribute
of the `CardiacTissue` object to an array of vectors with
`fibers.shape == mesh.shape + (dim,)` that represent the orientation of
fibers at each node in the mesh.
For 2D simulations, the fibers array should have shape `(n, n, 2)`.
For 3D simulations, the fibers array should have shape `(n, n, n, 3)`.

To use the fibers in the simulation, set the `stencil` attribute of the
tissue to `AsymmetricStencil2D` or `AsymmetricStencil3D`.
Using the `AsymmetricStencil` is more time-consuming than the `IsotropicStencil`.
For example, `AsymmetricStencil2D` requires 9-point stencil calculations vs.
5-point stencil calculations for `IsotropicStencil`.

To set diffusion coefficients in the direction of fibers, use the
`D_al` and `D_ac` attributes of the tissue. For Aliev-Panfilov model,
`D_al = 1` and `D_ac = D_al / 9`.

## Models

Finitewave provides several built-in models for simulating cardiac
electrophysiology:

- Aliev-Panfilov model
- TP06 model
- Luo-Rudy91 model

These models can be used to simulate the propagation of electrical waves in
cardiac tissue and study the effects of different conditions on wave
propagation.

## Stimulation

Stimulation is an essential aspect of cardiac electrophysiology simulations.
Finitewave provides several classes for defining and applying stimulation to
the tissue:

- `StimVoltageCoord2D`: Stimulate a region of the tissue with a voltage.
- `StimCurrentCoord2D`: Stimulate a region of the tissue with a current.
- `StimVoltageMatrix2D`: Stimulate the tissue with a matrix with the same
  shape as the `tissue.mesh`. The stimulation will be applied to the cells
  with non-zero values.
- `StimCurrentMatrix2D`: Stimulate the tissue with a matrix with the same
  shape as the `tissue.mesh`. The stimulation current will be applied to
  the cells with non-zero values.

Stimulation can be applied to the tissue using the `StimSequence` class,
which allows users to define a sequence of stimulations to be applied at
specific times during the simulation.


## Trackers

Trackers are used to measure various aspects of the simulation, such as
local activation times (LATs), electrogram signals (EGMs), and action
potentials, etc. Finitewave provides several built-in trackers for
measuring these properties:

- `ActivationTimeTracker`: Measure local activation times (LATs) in the tissue.
- `LocalActivationTimeTracker`: Measure multiple activation times in case of
  multiple stimuli or multiple waves.
- `ECGTracker`: Measure EGMs and ECGs signals in the tissue.
- `ActionPotentialTracker`: Measure action potentials in the specific cell
  of the tissue.
- `VariableTracker`: Measure any variable in the specific cell of the tissue.
- `AnimationTracker`: Create an animation using variable values in the tissue.
- `PeriodTracker`: Measure the period of the wave in the tissue.


## Visualization

Finitewave provides several tools for visualizing the results of cardiac
electrophysiology simulations in 3D. To show the results of the simulation,
use the `VisMeshBuilder3D` class to build a 3D mesh from the simulation
results and visualize the results using the `plot()` method. The
`VisMeshBuilder3D` based on `pyvista` library. The result can be saved as
`.vtk` or `.vtu` file for further analysis.

```python

   import finitewave as fw

   # Create a 3D mesh builder
   mesh_builder = fw.VisMeshBuilder3D()

   # Build a 3D mesh and mask empty (0) nodes
   grid = mesh_builder.build_mesh(tissue.mesh)

   # Add scalar data to the mesh
   grid = mesh_builder.add_scalar(aliev_panfilov.u, 'u')
   
   # Plot the mesh
   grid.plot(clim=[0, 1], cmap='viridis')

   # Save the potentials
   grid.save('u.vtk')

```