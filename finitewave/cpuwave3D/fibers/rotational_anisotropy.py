import numpy as np
import math

class RotationalAnisotropy:
    """
    A class to generate fiber orientations in a 3D space based on rotational anisotropy.

    Attributes
    ----------
    size : list of int
        A list defining the dimensions of the 3D grid (x, y, z).
    alpha : list of float
        A list with two elements defining the range of rotation angles in degrees.
    axis : int
        The axis along which the rotation will be applied (0 for x, 1 for y, 2 for z).
    init_v : list of float
        The initial vector along which the fibers are oriented.

    Methods
    -------
    generate_fibers():
        Generates and returns a 3D array of fibers with rotational anisotropy applied.
    test(fibers):
        Visualizes the generated fibers using a 3D quiver plot.
    """

    def __init__(self):
        """
        Initializes the RotationalAnisotropy object with default values.
        """
        self.size  = [0, 0, 0]
        self.alpha = [0, 0]
        self.axis = 0
        self.init_v = [1, 0, 0]

    def generate_fibers(self):
        """
        Generates a 3D array of fiber orientations based on rotational anisotropy.

        The fibers are initially aligned along `init_v` and then rotated according to the 
        specified range of angles (`alpha`) along the specified axis (`axis`).

        Returns
        -------
        numpy.ndarray
            A 4D NumPy array of shape (size[0], size[1], size[2], 3) containing the fiber vectors.
        """
        fibers = np.zeros(list(self.size) + [3])
        fibers[:,:,:,0] = self.init_v[0]
        fibers[:,:,:,1] = self.init_v[1]
        fibers[:,:,:,2] = self.init_v[2]
        alpha_step = (self.alpha[1] - self.alpha[0])/self.size[self.axis]

        c_fibers = np.copy(fibers)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    c = i
                    v = [1, 2]
                    if self.axis == 1:
                        c = j
                        v = [0, 2]
                    elif self.axis == 2:
                        c = k
                        v = [0, 1]

                    ang = math.radians(self.alpha[0] + alpha_step*c)

                    fibers[i, j, k, v[0]] = c_fibers[i, j, k, v[0]]*math.cos(ang) + c_fibers[i, j, k, v[1]]*math.sin(ang)
                    fibers[i, j, k, v[1]] = -c_fibers[i, j, k, v[0]]*math.sin(ang) + c_fibers[i, j, k, v[1]]*math.cos(ang)

                    norm = math.sqrt(fibers[i, j, k, 0]**2 + fibers[i, j, k, 1]**2 + fibers[i, j, k, 2]**2)

                    fibers[i, j, k, 0] /= norm
                    fibers[i, j, k, 1] /= norm
                    fibers[i, j, k, 2] /= norm

        return fibers

    def test(self, fibers):
        """
        Visualizes the generated fibers using a 3D quiver plot.

        Parameters
        ----------
        fibers : numpy.ndarray
            A 4D NumPy array of shape (size[0], size[1], size[2], 3) containing the fiber vectors.
        """
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = np.meshgrid(np.arange(0, self.size[1], 1),
                              np.arange(0, self.size[0], 1),
                              np.arange(0, self.size[2], 1))

        ax.quiver(x, y, z, fibers[:,:,:,1], fibers[:,:,:,0], fibers[:,:,:,2], length=1.0)

        plt.show()
