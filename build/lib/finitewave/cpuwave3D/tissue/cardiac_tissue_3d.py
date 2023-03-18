import numpy as np

from finitewave.core.tissue.cardiac_tissue import CardiacTissue
from finitewave.cpuwave3D.stencil.isotropic_stencil_3d import IsotropicStencil3D


class CardiacTissue3D(CardiacTissue):
    def __init__(self, shape, mode='iso'):
        CardiacTissue.__init__(self)
        self.meta["Dim"] = 3
        self.shape = shape
        self.mesh = np.ones(shape)
        self.add_boundaries()
        self.mode = mode
        self.stencil = IsotropicStencil3D()
        self.conductivity = 1
        self.fibers = None

    def add_boundaries(self):
        self.mesh[0, :, :] = 0
        self.mesh[:, 0, :] = 0
        self.mesh[:, :, 0] = 0
        self.mesh[-1, :, :] = 0
        self.mesh[:, -1, :] = 0
        self.mesh[:, :, -1] = 0

    def compute_weights(self, dr, dt):
        self.weights = self.stencil.get_weights(self.mesh, self.conductivity,
                                                self.fibers, self.D_al,
                                                self.D_ac, dt, dr)

    # def aniso_weights(self, dr, dt, D_al, D_ac):
    #     indexes = [1, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19,
    #                21, 22, 23, 25]
    #
    #     index_map = dict(zip(range(len(indexes)), indexes))
    #     weights = np.zeros((*self.shape, len(indexes)))
    #     self._compute_diffuse(D_al, D_ac)
    #
    #     from tqdm import tqdm
    #
    #     pbar = tqdm(total=(self.shape[0] - 2) * (self.shape[1] - 2) *
    #                       (self.shape[2] - 2))
    #
    #     for i in range(1, self.shape[0] - 1):
    #         for j in range(1, self.shape[1] - 1):
    #             for k in range(1, self.shape[2] - 1):
    #                 mesh_local = self.mesh[i-1: i+2, j-1: j+2, k-1: k+2]
    #                 mesh_local[mesh_local != 1] = 0
    #                 mesh_local = mesh_local.astype('int')
    #                 pbar.update(1)
    #
    #                 empty_center = mesh_local[1, 1, 1] != 1
    #                 isolated_center = np.sum(mesh_local) < 2
    #                 if empty_center or isolated_center:
    #                     continue
    #
    #                 diffuse = self.diffuse[i-1: i+2, j-1: j+2, k-1: k+2, :]
    #                 local_weights = self.stencil.get_weights(mesh_local, diffuse, dt,
    #                                                          dr).flatten()
    #
    #                 for ind in range(weights.shape[3]):
    #                     weights[i, j, k, ind] = local_weights[index_map[ind]]
    #     pbar.close()
    #     return weights
    #
    # def set_dtype(self, dtype):
    #     self.weights = self.weights.astype(dtype)
    #     self.mesh = self.mesh.astype(dtype)
    #
    # def _compute_diffuse(self, D_al, D_ac):
    #     self.diffuse[:, :, :, 0] = ((D_ac + (D_al - D_ac) *
    #                                  self.fibers[:, :, :, 0]**2) *
    #                                 self.conductivity)
    #     self.diffuse[:, :, :, 1] = (0.5 * (D_al - D_ac) *
    #                                 self.fibers[:, :, :, 0] *
    #                                 self.fibers[:, :, :, 1] *
    #                                 self.conductivity)
    #     self.diffuse[:, :, :, 2] = (0.5 * (D_al - D_ac) *
    #                                 self.fibers[:, :, :, 0] *
    #                                 self.fibers[:, :, :, 2] *
    #                                 self.conductivity)
    #     self.diffuse[:, :, :, 3] = ((D_ac + (D_al - D_ac) *
    #                                  self.fibers[:, :, :, 1]**2) *
    #                                 self.conductivity)
    #     self.diffuse[:, :, :, 4] = (0.5 * (D_al - D_ac) *
    #                                 self.fibers[:, :, :, 1] *
    #                                 self.fibers[:, :, :, 2] *
    #                                 self.conductivity)
    #     self.diffuse[:, :, :, 5] = ((D_ac + (D_al - D_ac) *
    #                                  self.fibers[:, :, :, 2]**2) *
    #                                 self.conductivity)
