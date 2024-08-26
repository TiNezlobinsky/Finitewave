# Finitewave

Package for a wide range of tasks in modeling cardiac electrophysiology using finite-difference methods.

## Package structure

*/finitewave*

The package src.

*/examples*

Scripts with the demonstration of different aspects of using finitewave package.

*/tests*

A set of tests to check the correctness of the finitewave package functionality.

## Installation

```sh
python -m build
pip install dist/finitewave-_version_.whl
```

or to make editable package:

```sh
pip install -e .
```

## Running examples

Go to the ./examples for more details.


## Running tests

To run tests use this command (example for Aliev Panfilov 2D model test):

```sh
python -m unittest test_aliev_panfilov_2d.py
```

## Requirements

| Dependency | Version\* | Link                        |
| ---------- | --------- | --------------------------- |
| numpy      | 1.26.4    | https://numpy.org           |
| numba      | 0.59.0    | https://numba.pydata.org    |
| scipy      | 1.11.4    | https://scipy.org           |
| matpltolib | 3.8.3     | https://matplotlib.org      |
| tqdm       | 4.65.0    | https://github.com/tqdm     |
| vtk        | 9.3.0     | https://vtk.org             |

*last tested version.

vtk is optional and you can pass installation if you are not going to visualize 3D meshes.

If you are going to use AnimationBuilder to create mp4 animations, please install the ffmpeg on your device.
