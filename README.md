# Finitewave

**Finitewave** is a Python package for simulating cardiac electrophysiology using finite-difference methods. It provides tools for modeling and visualizing the propagation of electrical waves in cardiac tissue, making it ideal for researchers and engineers in computational biology, bioengineering, and related fields.
<p align="center">
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/images/wave_2d.gif" height="200" width="200" />
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/images/spiral_wave_2d.gif" height="200" width="267" />
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/images/spiral_wave_3d.gif" height="200" width="220" />
</p>

### Why Finitewave? 

Because of its simplicity and availability. Finitewave is the most simple and user-friendly framework for cardiac simulation, supporting a rich set of tools that make it accessible to both beginners and advanced users alike.

## Features

- Simulate 2D and 3D cardiac tissue models, including the ability to handle complex geometries.
- Simulate conditions such as fibrosis and infarction. 
- Built-in models, including the Aliev-Panfilov, TP06, Luo-Rudy91 models.
- Trackers for measuring various aspects of the simulation (such as activation time or EGMs) 
- Visualization tools for analyzing wave propagation.
- Customize simulation parameters to suit specific research needs.
- High-performance computing with support for GPU acceleration (currently under development).

## Installation

To install Finitewave, navigate to the root directory of the project and run:

```sh
python -m build
pip install dist/finitewave-<version>.whl
```

This will install Finitewave as a Python package on your system.

For development purposes, you can install the package in an editable mode, which allows changes to be immediately reflected without reinstallation:

```sh
pip install -e .
```

## Requirements

| Dependency | Version\* | Link                        |
| ---------- | --------- | --------------------------- |
| numpy      | 1.26.4    | https://numpy.org           |
| numba      | 0.59.0    | https://numba.pydata.org    |
| scipy      | 1.11.4    | https://scipy.org           |
| matplotlib | 3.8.3     | https://matplotlib.org      |
| tqdm       | 4.65.0    | https://github.com/tqdm     |
| pyvista    | 0.44.1    | https://pyvista.org         |

*Versions listed are the most recent tested versions.

If you want to use the AnimationBuilder to create MP4 animations,
ensure that ffmpeg is installed on your system.

## Quick Start

Here's a simple example to get you started:

```python
import finitewave as fw

n = 100

# Initialize a 100x100 mesh with all nodes set to 1 (1 = cardiomyocytes, healthy cardiac tissue)
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n]) 
tissue.add_boundaries() # Add empty nodes (0) at the mesh edges

# Use Aliev-Panfilov model to perform simulation
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01  # time step
aliev_panfilov.dr = 0.25  # space step
aliev_panfilov.t_max = 5  # simulation time

# Set up stimulation parameters (activation from a line of nodes in the mesh)
stim_sequence = fw.StimSequence()
# activation time, activation value (voltage model values), stimulation area geometry - line with length n and width 3 (0, n, 0, 3)  
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, 3))
# Assign the tissue and stimulation parameters to the model
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# Display the potential map at the end of the simulation
plt.imshow(aliev_panfilov.u)
plt.show()
```

## Minimal script requirements

To create a simulation script using Finitewave, ensure you include the following minimal set of components:

CardiacTissue:
-> Set up the mesh, fibers array, stencil, and conductivity array.
- `mesh`: Ensure that every mesh contains a border line of empty nodes (boundary). Use the add_boundaries() method to easily add these boundary nodes.
- `stencil`: Choose between a 9-point stencil (anisotropic) or a 5-point stencil (orthotropic or isotropic). The stencil calculates weights for the divergence kernels. While the 9-point stencil is general-purpose, using the 5-point stencil is more performance-efficient in orthotropic and isotropic diffusion cases.
- `conductivity`: This array of coefficients (default: 1) to simulate propagation speed. This is the simplest (but not the only) way to model fibrotic tissue.

Model Setup:
- Create and configure the model with a minimal set of parameters: **dt** (time step), **dr** (spatial step), and **t_max** (maximum simulation time).

Stimulation Parameters:
- Use `Stim` classes to define the stimulation area and add them to the StimSequence class object. For example (for 2D simulations):
- - `StimVoltageCoord2D`: [stim_time, voltage, x0, x1, y0, y1]
- - `StimCurrentCoord2D`: [stim_time, current, current_time, x0, x1, y0, y1]
- Run the simulation using the `run()` method.

## Quick Tutorial

For detailed information and practical examples, please refer to the `examples/` folder.

Currently, we explicitly use 2D and 3D versions of the Finitewave class objects. This means that most of the classes you encounter in the scripts will have either `2D` or `3D` appended to their names.

### Cardiac Tissue

The `CardiacTissue` class is used to represent myocardial tissue and its structural properties in simulations. It includes several key attributes that define the characteristics and behavior of the cardiac mesh used in finite-difference calculations.

#### Mesh

The `mesh` attribute is a finite-difference mesh consisting of nodes, which represent the myocardial structure. The distance between neighboring nodes is defined by the spatial step (**dr**) parameter of the model. The nodes in the mesh are used to represent different types of tissue and their properties:

- `0`: Empty node, representing the absence of cardiac tissue.
- `1`: Healthy cardiac tissue, which supports wave propagation.
- `2`: Fibrotic or infarcted tissue, representing damaged or non-conductive areas.
Nodes marked as `0` and `2` are treated similarly as isolated nodes with no flux through their boundaries. These different notations help distinguish between areas of healthy tissue, empty spaces, and regions of fibrosis or infarction.

To satisfy boundary conditions, every Finitewave mesh must include boundary nodes (marked as `0`). This can be easily achieved using the `add_boundaries()` method, which automatically adds rows of empty nodes around the edges of the mesh.

You can also utilize `0` nodes to define complex geometries and pathways, or to model organ-level structures. For example, to simulate the electrophysiological activity of the heart, you can create a 3D array where `1` represents cardiac tissue, and `0` represents everything outside of that geometry.

#### Fibers

Another important attribute, `fibers`, is used to define the anisotropic properties of cardiac tissue. This attribute is represented as a 3D array (for 2D tissue) or a 4D array (for 3D tissue), with each node containing a 2D or 3D vector that specifies the fiber orientation at that specific position. The anisotropic properties of cardiac tissue mean that the wave propagation speed varies depending on the fiber orientation. Typically, the wave speed is three times faster along the fibers compared to across the fibers, which can be set by adjusting the diffusion coefficients ratio (**D_al/D_ac**) to 9.

#### Conductivity

The conductivity attribute defines the local conductivity of the tissue and is represented as an array of coefficients ranging from **0.0** to **1.0** for each node in the mesh. It proportionally decreases the diffusion coefficient locally, thereby slowing down the wave propagation in specific areas defined by the user. This is useful for modeling heterogeneous tissue properties, such as regions of impaired conduction due to ischemia or fibrosis.

### Built-in Models

Finitewave currently includes three built-in models for 2D and 3D simulations. Each model represents the cardiac electrophysiological activity of a single cell, which can be combined using parabolic equations to form complex 2D or 3D cardiac tissue models.

We use an explicit finite-difference scheme, which requires maintaining an appropriate **dt/dr** ratio. The recommended calculation parameters for time and space steps are **dt** = 0.01 and **dr** = 0.25. You can increase **dt** to 0.02 to speed up calculations, but always verify the stability of your numerical scheme, as instability will lead to incorrect simulation results.

| Model          | Description                                                   | 
| -------------- | ------------------------------------------------------------- | 
| Aliev-Panfilov | A phenomenological two-variable model for cardiac simulations  |
| Luo-Rudy 1991  | An ionic model for cardiac simulations                         | 
| TP06           | An ionic model for cardiac simulations                         |

### Trackers

Trackers are one of the key features of Finitewave. They allow you to measure a wide range of data during a simulation. Multiple trackers can be used simultaneously by adding them to the `TrackerSequence` class.

Here is a list of the currently implemented trackers:

| Tracker               | Description                                                                   | 
| --------------------- | ----------------------------------------------------------------------------- | 
| Activation Time       | Measures the time of the first wave arrival at each mesh node.                |
| Animation             | Creates model snapshots of selected variables, which can be used to build animations. | 
| ECG                   | Measures ECG at specific positions.                                            |
| Multi-Activation Time | Measures the time of multiple wave arrivals at each mesh node.                |
| Multi-Variable        | Measures the dynamics of variables at specific nodes.                         |
| Period                | Measures the period of wave dynamics (e.g., spiral waves).                    |
| Period Map            | Measures period dynamics at mesh nodes and creates snapshots.                 |
| Tips                  | Tracks spiral wave tip trajectories.                                          | 
| Velocity              | Measures the velocity of planar waves.                                        |

### Stimulations

There are two basic options to stimulate electrophysiological activity in cardiac tissue using Finitewave. 

1. **Voltage Stimulation**: This method directly sets voltage values at the nodes within the stimulation area, triggering wave propagation from this region.
2. **Current Stimulation**: In this method, you apply a current value and stimulation duration to accumulate potential, leading to wave propagation. Current stimulation offers more flexibility and is more physiologically accurate, as it simulates the activity of external electrodes.

An important parameter is the **area of stimulation**. You can choose between a simple rectangular stimulation class (as shown in the Quick Start section) or a flexible matrix stimulation that allows you to define stimulation areas as a Boolean array, where `True` values indicate nodes to be stimulated.

You can simulate a sequence of stimulations (e.g., a high-pacing protocol) by adding multiple stimulations to the `StimulationTracker` class.

**Note**: A very small stimulation area may lead to unsuccessful stimulation due to a source-sink mismatch.


## Package structure

*/finitewave*

Core package source code.

*/examples*

Scripts demonstrating various functionalities of the Finitewave package.

*/tests*

Unit tests to ensure the correctness and reliability of the package.

## Running tests

To run tests, you can use the following command, for example, to test the 2D Aliev-Panfilov model:

```sh
python -m unittest test_aliev_panfilov_2d.py
```

## Contribution

Contributions are welcome!

### How to Contribute
- Fork the repository
- Create a new branch (`git checkout -b feature-branch`)
- Commit your changes (`git commit -m 'Add new feature'`)
- Push to the branch (`git push origin feature-branch`)
- Open a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.
