import numpy as np
from tqdm import tqdm

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave3D.model.tp06_3d.tp06_kernels_3d import \
    TP06Kernels3D

_npfloat = "float64"


class TP063D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)
        self.D_al = 0.154
        self.D_ac = 0.154
        self.m = np.ndarray
        self.h = np.ndarray
        self.j_ = np.ndarray
        self.d = np.ndarray
        self.f = np.ndarray
        self.x = np.ndarray
        self.Cai_c = np.ndarray
        self.model_parameters = {}
        self.state_vars = ["u", "Cai", "CaSR", "CaSS", "Nai", "Ki",
                           "M_", "H_", "J_", "Xr1", "Xr2", "Xs", "R_",
                           "S_", "D_", "F_", "F2_", "FCass", "RR", "OO"]
        self.npfloat = 'float64'

    def initialize(self):
        super().initialize()
        weights_shape = self.cardiac_tissue.weights.shape
        shape = self.cardiac_tissue.mesh.shape
        self.kernel_diffuse = TP06Kernels3D().get_diffuse_kernel(weights_shape)
        self.kernel_vars = TP06Kernels3D().get_ionic_kernel()

        self.u = -84.5*np.ones(shape, dtype=_npfloat)
        self.u_new = self.u.copy()
        self.Cai = 0.00007*np.ones(shape, dtype=_npfloat)
        self.CaSR = 1.3*np.ones(shape, dtype=_npfloat)
        self.CaSS = 0.00007*np.ones(shape, dtype=_npfloat)
        self.Nai = 7.67*np.ones(shape, dtype=_npfloat)
        self.Ki = 138.3*np.ones(shape, dtype=_npfloat)
        self.M_ = np.zeros(shape, dtype=_npfloat)
        self.H_ = 0.75*np.ones(shape, dtype=_npfloat)
        self.J_ = 0.75*np.ones(shape, dtype=_npfloat)
        self.Xr1 = np.zeros(shape, dtype=_npfloat)
        self.Xr2 = np.ones(shape, dtype=_npfloat)
        self.Xs = np.zeros(shape, dtype=_npfloat)
        self.R_ = np.zeros(shape, dtype=_npfloat)
        self.S_ = np.ones(shape, dtype=_npfloat)
        self.D_ = np.zeros(shape, dtype=_npfloat)
        self.F_ = np.ones(shape, dtype=_npfloat)
        self.F2_ = np.ones(shape, dtype=_npfloat)
        self.FCass = np.ones(shape, dtype=_npfloat)
        self.RR = np.ones(shape, dtype=_npfloat)
        self.OO = np.zeros(shape, dtype=_npfloat)

    def run_ionic_kernel(self):
        self.ionic_kernel(self.u_new, self.u, self.Cai, self.CaSR, self.CaSS,
                          elf.Nai, self.Ki, self.M_, self.H_, self.J_, self.Xr1,
                          self.Xr2, self.Xs, self.R_, self.S_, self.D_, self.F_,
                          self.F2_, self.FCass, self.RR, self.OO,
                          self.cardiac_tissue.mesh, self.dt)
