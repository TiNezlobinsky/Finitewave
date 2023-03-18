import numpy as np
from tqdm import tqdm

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave3D.model.luo_rudy91_3d.luo_rudy91_kernels_3d import \
    LuoRudy91Kernels3D

_parallel = True
_npfloat = "float64"


class LuoRudy913D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)
        self.D_al = 0.1
        self.D_ac = 0.1
        self.I_tot = np.ndarray
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.model_parameters = {}
        self.state_vars = ["u", "m", "h", "j_", "d", "f", "x", "Cai_c"]
        self.npfloat = 'float64'
        self.parallel = 'True'

    def initialize(self):
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape
        self.kernel_diffuse = LuoRudy91Kernels3D().get_diffuse_kernel(weights_shape)
        self.kernel_vars = LuoRudy91Kernels3D().get_ionic_kernel()

        self.u = -84.5*np.ones(shape, dtype=_npfloat)
        self.u_new = self.u.copy()
        self.m = 0.0017*np.ones(shape, dtype=_npfloat)
        self.h = 0.9832*np.ones(shape, dtype=_npfloat)
        self.j_ = 0.995484*np.ones(shape, dtype=_npfloat)
        self.d = 0.000003*np.ones(shape, dtype=_npfloat)
        self.f = np.ones(shape, dtype=_npfloat)
        self.x = 0.0057*np.ones(shape, dtype=_npfloat)
        self.Cai_c = 0.0002*np.ones(shape, dtype=_npfloat)
        self.I_tot = np.zeros(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        self.ionic_kernel(self.u_new, self.u, self.m, self.h, self.j_, self.d,
                          self.f, self.x, self.Cai_c, self.cardiac_tissue.mesh,
                          self.dt)
