import numpy as np

from finitewave.core.tissue.cardiac_tissue import CardiacTissue
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import IsotropicStencil2D


class CardiacTissue2D(CardiacTissue):
    def __init__(self, shape, mode='iso'):
        CardiacTissue.__init__(self)
        self.meta["Dim"] = 2
        self.shape = shape
        self.mesh = np.ones(shape)
        self.add_boundaries()
        self.stencil = IsotropicStencil2D()
        self.conductivity = 1
        self.fibers = None

    def add_boundaries(self):
        self.mesh[0, :] = 0
        self.mesh[:, 0] = 0
        self.mesh[-1, :] = 0
        self.mesh[:, -1] = 0

    def compute_weights(self, dr, dt):
        self.weights = self.stencil.get_weights(self.mesh, self.conductivity,
                                                self.fibers, self.D_al,
                                                self.D_ac, dt, dr)
        # if self.mode == 'iso':
        #     self.weights = self.stencil.get_weights(self.mesh,
        #                                             D_al*self.conductivity,
        #                                             dt,
        #                                             dr)
        # elif self.mode == 'aniso':
        #     self.weights = self.aniso_weights(dr, dt, D_al, D_ac)
        # else:
        #     raise IncorrectWeightsModeError2D()

    # def aniso_weights(self, dr, dt, D_al, D_ac):
    #     indexes = list(range(9))
    #
    #     index_map = dict(zip(range(len(indexes)), indexes))
    #     weights = np.zeros((*self.shape, len(indexes)))
    #     self._compute_diffuse(D_al, D_ac)
    #
    #     for i in range(1, self.shape[0] - 1):
    #         for j in range(1, self.shape[1] - 1):
    #             mesh_local = self.mesh[i-1: i+2, j-1: j+2]
    #             mesh_local[mesh_local != 1] = 0
    #             mesh_local = mesh_local.astype('int')
    #
    #             empty_center = mesh_local[1, 1] != 1
    #             isolated_center = np.sum(mesh_local) < 2
    #             if empty_center or isolated_center:
    #                 continue
    #
    #             diffuse = self.diffuse[i-1: i+2, j-1: j+2, :]
    #             local_weights = self.stencil.get_weights(mesh_local, diffuse, dt,
    #                                                      dr).flatten()
    #
    #             for ind in range(weights.shape[2]):
    #                 weights[i, j, ind] = local_weights[index_map[ind]]
    #     return weights

    # def set_dtype(self, dtype):
    #     self.weights = self.weights.astype(dtype)
    #     self.mesh = self.mesh.astype(dtype)

    # def _compute_diffuse(self, D_al, D_ac):
    #     self.diffuse[:, :, 0] = ((D_ac + (D_al - D_ac) *
    #                              self.fibers[:, :, 0]**2) * self.conductivity)
    #     self.diffuse[:, :, 1] = (0.5 * (D_al - D_ac) *
    #                              self.fibers[:, :, 0] * self.fibers[:, :, 1] *
    #                              self.conductivity)
    #     self.diffuse[:, :, 2] = ((D_ac + (D_al - D_ac) *
    #                              self.fibers[:, :, 1]**2) * self.conductivity)
    #
    #     fibers_x = self.fibers[:-1, :, :] + self.fibers[1:, :, :]
    #     fibers_x = fibers_x / np.linalg.norm(fibers_x, axis=2)[:, :, np.newaxis]
    #     fibers_y = self.fibers[:, :-1, :] + self.fibers[:, 1:, :]
    #     fibers_y = fibers_y / np.linalg.norm(fibers_y, axis=2)[:, :, np.newaxis]
