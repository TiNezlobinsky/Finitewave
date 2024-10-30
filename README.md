# Finitewave

**Finitewave** is a Python package for simulating cardiac electrophysiology using finite-difference methods.
It provides tools for modeling and visualizing the propagation of electrical waves in cardiac tissue,
making it ideal for researchers and engineers in computational biology, bioengineering, and related fields.
<p align="center">
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/docs/wave_2d.gif" height="200" width="200" />
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/docs/spiral_wave_2d.gif" height="200" width="267" />
  <img src="https://github.com/TiNezlobinsky/Finitewave/blob/main/docs/spiral_wave_3d.gif" height="200" width="220" />
</p>

You can find more information on our website [not a link yet]()

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
