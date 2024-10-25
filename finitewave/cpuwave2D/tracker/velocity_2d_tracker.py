import os
import numpy as np
from scipy.spatial.distance import euclidean
import json

from finitewave.cpuwave2D.tracker.activation_time_2d_tracker import ActivationTime2DTracker


def _local_velocity(act_t):
    """
    Calculates local velocities based on activation times.

    Parameters
    ----------
    act_t : numpy.ndarray
        2D array of activation times.

    Returns
    -------
    numpy.ndarray
        2D array of local velocities.
    """
    N1, N2 = act_t.shape
    vel = np.zeros(act_t.shape)
    for i in range(1, N1-1):
        for j in range(1, N2-1):
            times = [act_t[i-1, j], act_t[i+1, j],
                     act_t[i, j-1], act_t[i, j+1]]
            if times:
                min_t = np.min(times)
                vel[i, j] = act_t[i, j] - min_t
    return vel


class Velocity2DTracker(ActivationTime2DTracker):
    """
    A tracker that calculates the front velocity of activation based on activation times
    from a 2D model. Inherits from `ActivationTime2DTracker`.

    Attributes
    ----------
    file_name : str
        Name of the file where the velocity data will be saved. Default is "front_velocity".
    """
    def __init__(self):
        super().__init__()
        self.file_name = "front_velocity"

    def initialize(self, model):
        """
        Initializes the tracker with the given model.

        Parameters
        ----------
        model : object
            The model object from which data is being tracked. It must have attributes
            `dr` (distance resolution) for computing velocities.
        """
        ActivationTime2DTracker.initialize(self, model)

    def compute_velocity_front(self):
        """
        Computes the front velocity of activation based on the activation times.

        Returns
        -------
        numpy.ndarray
            2D array of velocities at the front of activation.
        """
        # all empty nodes are -1
        # initial activation nodes are 0
        act_t = self.act_t
        dr    = self.model.dr

        max_act = np.max(act_t)
        front_vel = np.zeros(act_t[act_t == max_act].shape)
        ind_front = np.argwhere(act_t == max_act)

        ind_stim  = np.argwhere(act_t == np.min(act_t[act_t >= 0.]))
        for i, ind_f in enumerate(ind_front):
            try:
                near_ind = np.argmin(np.array(list(map(
                                     lambda sp: (sp[0] - ind_f[0])**2 + (sp[1] - ind_f[1])**2,
                                     ind_stim))))
            except ValueError:
                continue
            front_vel[i] = euclidean(ind_stim[near_ind], ind_f)*dr/max_act
        return front_vel

    @property
    def output(self):
        """
        Computes and returns the front velocity of activation.

        Returns
        -------
        numpy.ndarray
            2D array of velocities at the front of activation.
        """
        return self.compute_velocity_front()

    def write(self):
        """
        Writes the computed front velocities to a JSON file.
        """
        jdata = json.dumps(self.compute_velocity_front())
        with open(self.file_name, "w") as jf:
            jf.write(jdata)
