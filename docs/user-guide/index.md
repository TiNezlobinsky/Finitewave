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

