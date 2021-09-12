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
python setup.py install
```

## Running examples

Go to the ./examples for more details.


## Running tests

To run the Aliev Panfilov 2D model test:

```sh
python -m unittest test_aliev_panfilov_2d.py
```

Other tests work in the same way.

## Requirements

- numpy
- numba
- scipy
- matpltolib
- tqdm
- vtk

If you are going to use AnimationBuilder to create mp4 animations, please install the ffmpeg on your device.
