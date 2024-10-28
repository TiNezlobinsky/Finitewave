import numpy as np
from scipy.spatial import distance
import json


class PlanarWaveVelocity2DCalculation:
    """
    Class for calculating the front velocity of activation in 2D.

    Attributes
    ----------

    """
    def __init__(self):
        self.file_name = "front_velocity"

    def compute_velocity_front(self, act_t, dr):
        """
        Computes the front velocity of activation based on the activation
        times.

        Parameters
        ----------
        act_t : numpy.ndarray
            2D array of activation times.
        dr : float
            Spatial resolution.

        Returns
        -------
        numpy.ndarray
            2D array of velocities at the front of activation.
        """
        # all empty nodes are -1
        # initial activation nodes are 0
        max_act = np.max(act_t)
        front_vel = np.zeros(act_t[act_t == max_act].shape)
        ind_front = np.argwhere(act_t == max_act)

        ind_stim = np.argwhere(act_t == np.min(act_t[act_t >= 0.]))
        for i, ind_f in enumerate(ind_front):
            try:
                near_ind = np.argmin(np.array(list(map(
                                     lambda sp: ((sp[0] - ind_f[0])**2
                                                 + (sp[1] - ind_f[1])**2),
                                     ind_stim))))
            except ValueError:
                continue
            front_vel[i] = distance.euclidean(ind_stim[near_ind],
                                              ind_f) * dr / max_act
        return front_vel

    def write(self):
        """
        Computes and writes the computed front velocities to a JSON file.
        """
        jdata = json.dumps(self.compute_velocity_front())
        with open(self.file_name, "w") as jf:
            jf.write(jdata)
