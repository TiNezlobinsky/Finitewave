from setuptools import setup, find_packages

setup(
    name="finitewave",
    description=("Simple finite-difference package for electrical cardiac"
                 " modeling tasks solution"),
    version="0.8",
    packages=find_packages(exclude=["examples", "tests"]),
    install_requires=["numpy", "scipy", "numba", "matplotlib",
                      "tables", "h5py", "tqdm", "vtk"]
)
