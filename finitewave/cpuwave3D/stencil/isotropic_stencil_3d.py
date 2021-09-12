import numbers
import numpy as np

from finitewave.core.stencil.stencil import Stencil


class IsotropicStencil3D(Stencil):
    def __init__(self):
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 7))

        diffuse = D_al * conductivity * np.ones(mesh.shape)

        weights[:, :, :, 0] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=0)
        weights[:, :, :, 1] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=1)
        weights[:, :, :, 2] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=2)
        weights[:, :, :, 4] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=2)
        weights[:, :, :, 5] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=1)
        weights[:, :, :, 6] = diffuse * dt / (dr**2) * np.roll(mesh, -1,
                                                               axis=0)

        # heterogeneity
        diff_i = np.roll(diffuse, 1, axis=0) - np.roll(diffuse, -1, axis=0)
        diff_j = np.roll(diffuse, 1, axis=1) - np.roll(diffuse, -1, axis=1)
        diff_k = np.roll(diffuse, 1, axis=2) - np.roll(diffuse, -1, axis=2)

        weights[:, :, :, 0] -= dt / (2*dr) * diff_i
        weights[:, :, :, 1] -= dt / (2*dr) * diff_j
        weights[:, :, :, 2] -= dt / (2*dr) * diff_k
        weights[:, :, :, 4] += dt / (2*dr) * diff_k
        weights[:, :, :, 5] += dt / (2*dr) * diff_j
        weights[:, :, :, 6] += dt / (2*dr) * diff_i

        for i in [0, 1, 2, 4, 5, 6]:
            weights[:, :, :, i] *= mesh
            weights[:, :, :, 3] -= weights[:, :, :, i]
        weights[:, :, :, 3] += 1
        weights[:, :, :, 3] *= mesh
        return weights
