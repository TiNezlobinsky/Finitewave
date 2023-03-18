import numbers
import numpy as np

from finitewave.core.stencil.stencil import Stencil


class IsotropicStencil2D(Stencil):
    def __init__(self):
        Stencil.__init__(self)

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        weights = np.zeros((*mesh.shape, 5))

        diffuse = D_al * conductivity * np.ones(mesh.shape)

        weights[:, :, 0] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=0)
        weights[:, :, 1] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=1)
        weights[:, :, 3] = diffuse * dt / (dr**2) * np.roll(mesh, -1, axis=1)
        weights[:, :, 4] = diffuse * dt / (dr**2) * np.roll(mesh, -1, axis=0)

        # heterogeneity
        diff_i = np.roll(diffuse, 1, axis=0) - np.roll(diffuse, -1, axis=0)
        diff_j = np.roll(diffuse, 1, axis=1) - np.roll(diffuse, -1, axis=1)

        weights[:, :, 0] -= dt / (2*dr) * diff_i
        weights[:, :, 1] -= dt / (2*dr) * diff_j
        weights[:, :, 3] += dt / (2*dr) * diff_j
        weights[:, :, 4] += dt / (2*dr) * diff_i

        for i in [0, 1, 3, 4]:
            weights[:, :, i] *= mesh
            weights[:, :, 2] -= weights[:, :, i]
        weights[:, :, 2] += 1
        weights[:, :, 2] *= mesh

        return weights
