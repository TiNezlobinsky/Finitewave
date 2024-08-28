import os
import numpy as np
from scipy.spatial.distance import euclidean
import json

from finitewave.cpuwave2D.tracker.activation_time_2d_tracker import ActivationTime2DTracker


def _local_velocity(act_t):
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
    def __init__(self):
        ActivationTime2DTracker.__init__(self)
        self.file_name = "front_velocity"

    def initialize(self, model):
        ActivationTime2DTracker.initialize(self, model)

    def compute_velocity_front(self):
        # all empty nodes are -1
        # intial activation nodes are 0
        act_t = self.act_t
        dr    = self.model.dr

        max_act = np.max(act_t)
        print (max_act)
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
        return self.compute_velocity_front()

    def write(self):
        jdata = json.dumps(self.compute_velocity_front())
        with open(self.file_name, "w") as jf:
            jf.write(jdata)
