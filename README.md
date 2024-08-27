# Finitewave

**Finitewave** is a Python package for simulating cardiac electrophysiology using finite-difference methods. It provides tools for modeling and visualizing the propagation of electrical waves in cardiac tissue, making it ideal for researchers and engineers in computational biology, bioengineering, and related fields.


## Features

- Simulate 2D and 3D cardiac tissue models.
- Built-in models, including the Aliev-Panfilov, TP06, Luo-Rudy91 models.
- Trackers for measuring various aspects of the simulation (such as activation time or EGMs) 
- Visualization tools for analyzing wave propagation.
- Customizable parameters for tailored simulations.
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
| matpltolib | 3.8.3     | https://matplotlib.org      |
| tqdm       | 4.65.0    | https://github.com/tqdm     |
| vtk**      | 9.3.0     | https://vtk.org             |

*Versions listed are the most recent tested versions.

**VTK is optional and not included in the default installation (pyproject.toml). Install it if you plan to visualize 3D meshes.

If you want to use the AnimationBuilder to create MP4 animations, ensure that ffmpeg is installed on your system.

## Quick Start

Here's a simple example to get you started:

```python
import numpy as np
import matplotlib.pyplot as plt

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

n = 100

# Initialize a 100x100 mesh with all nodes set to 1 (1 = cardiomyocytes, healthy cardiac tissue)
tissue = CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n]) 
tissue.add_boundaries() # Add empty nodes (0) at the mesh edges

# Use Aliev-Panfilov model to perform simulation
aliev_panfilov = AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01  # time step
aliev_panfilov.dr = 0.25  # space step
aliev_panfilov.t_max = 5  # simulation time

# Set up stimulation parameters (activation from a line of nodes in the mesh)
stim_sequence = StimSequence()
# activation time, activation value (voltage model values), stimulation area geometry - line with length n and width 3 (0, n, 0, 3)  
stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, n, 0, 3))
# Assign the tissue and stimulation parameters to the model
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# Display the potential map at the end of the simulation
plt.imshow(aliev_panfilov.u)
plt.show()
```

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