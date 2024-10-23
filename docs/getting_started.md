## Getting Started

Here's a simple example to get you started:

```python

   import finitewave as fw

   n = 100

   # Initialize a 100x100 simulation mesh
   tissue = fw.CardiacTissue2D([n, n])
   tissue.mesh = np.ones([n, n]) # Set all nodes to 1 (cardiomyocytes)
   tissue.add_boundaries() # Make the border nodes empty

   # Use Aliev-Panfilov model to perform simulation
   model = fw.AlievPanfilov2D()
   # Setup numerical parameters:
   model.dt = 0.01  # time step
   model.dr = 0.25  # spatial step
   model.t_max = 5  # simulation time

   # Setup stimulation
   stim_sequence = fw.StimSequence()
   # Set one stimul (u = 1) at the beginning of the simulation on the edge of 
   # the tissue (line with length n and width 3)  
   stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 3))
   # Set the tissue and stimulation sequence to the model
   model.cardiac_tissue = tissue
   model.stim_sequence = stim_sequence

   model.run()

   # Display the potential map at the end of the simulation
   plt.imshow(model.u)
   plt.show()

```

## Minimal script requirements

To create a simulation script using Finitewave, ensure you include the
following minimal set of components:

### CardiacTissue

Set up the mesh, fibers array, stencil, and conductivity array.

- `mesh`: Mesh contains cells (nodes) that represent the cardiac tissue.
  Each cell can be assigned a value to represent the type of tissue
  (cardiomyocytes, fibrotic tissue, and boundary). Ensure that every mesh
  contains a border line of empty nodes (boundary) by using `add_boundaries()`
  method.

- `stencil`: Choose between a 9-point stencil (anisotropic) or a 5-point
  stencil (orthotropic or isotropic). The stencil calculates weights for the
  divergence kernels. While the 9-point stencil is general-purpose, using
  the 5-point stencil is more performance-efficient in orthotropic and
  isotropic diffusion cases.

- `conductivity`: This array of coefficients (default: 1) to simulate
  propagation speed. This is the simpliest (but not the only) way to model
  fibrotic tissue.

### Model Configuration

Create and configure the model with a minimal set of parameters:

- `dt` (time step)
- `dr` (spatial step)
- `t_max` (maximum simulation time).

### Stimulation Parameters

Use `Stim` classes to define the stimulation area and add them to the
`StimSequence` class object.

For example (for 2D simulations):

- `StimVoltageCoord2D(stim_time, voltage, x0, x1, y0, y1)`
- `StimCurrentCoord2D(stim_time, current, current_time, x0, x1, y0, y1)`
- Run the simulation using the `run()` method.