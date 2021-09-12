import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from finitewave.cpuwave2D.model.diffuse_kernels_2d \
    import diffuse_kernel_2d_iso, diffuse_kernel_2d_aniso, _parallel


class AlievPanfilovKernels2D:
    def __init__(self):
        pass

    def get_diffuse_kernel()
