import numpy as np
import math


class RotationalAnisotropy:
    def __init__(self):
        self.size  = [0, 0, 0]
        self.alpha = [0, 0]
        self.axis = 0
        self.init_v = [1, 0, 0]

    def generate_fibers(self):
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

                    # if i == 1 and k == 1:
                    #     print (math.degrees(ang))

        # self.test(fibers)

        return fibers

    def test(self, fibers):

        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = np.meshgrid(np.arange(0, self.size[1], 1),
                              np.arange(0, self.size[0], 1),
                              np.arange(0, self.size[2], 1))

        ax.quiver(x, y, z, fibers[:,:,:,1], fibers[:,:,:,0], fibers[:,:,:,2], length=1.0)

        plt.show()
