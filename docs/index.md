# Finitewave

**Finitewave** is a Python package for simulating cardiac electrophysiology
using finite-difference methods. It provides tools for modeling and visualizing
the propagation of electrical waves in cardiac tissue, making it ideal for
researchers and engineers in computational biology, bioengineering,
and related fields.

## Why Finitewave?

Because of its simplicity and availability. Finitewave is the most simple and
user-friendly framework for cardiac simulation, supporting a rich set of tools
that make it accessible to both beginners and advanced users alike.

### Features

- Simulate 2D and 3D cardiac tissue models, including the ability to handle
  complex geometries.
- Simulate conditions such as fibrosis and infarction. 
- Built-in models, including the Aliev-Panfilov, TP06, Luo-Rudy91 models.
- Trackers for measuring various aspects of the simulation (LATs, EGMs, Action
  potentials, etc.). 
- Visualization tools for analyzing wave propagation.
- Customize simulation parameters to suit specific research needs.
- High-performance computing with support for GPU acceleration (currently under
  development).

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

If you want to use the `AnimationBuilder` to create MP4 animations,
ensure that ffmpeg is installed on your system.
