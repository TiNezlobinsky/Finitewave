import os
import numpy as np
from scipy.spatial.distance import euclidean
import json

from finitewave.cpuwave3D.tracker.activation_time_3d_tracker import ActivationTime3DTracker


class Velocity3DTracker(ActivationTime3DTracker):
    def __init__(self):
        ActivationTime3DTracker.__init__(self)
        self.file_name = "front_velocity"

    def initialize(self, model):
        ActivationTime3DTracker.initialize(self, model)

    def compute_velocity_front(self):
        # all empty nodes are -1
        # intial activation nodes are 0
        act_t = self.act_t
        dr    = self.model.dr

        max_act = np.max(act_t)
        # front_vel = np.zeros(act_t[act_t == max_act].shape)
        ind_front = np.argwhere(act_t == max_act)
        ind_front_i = np.mean(ind_front[:, 0])
        ind_front_j = np.mean(ind_front[:, 1])
        ind_front_k = np.mean(ind_front[:, 2])

        ind_stim  = np.argwhere(act_t == np.min(act_t[act_t >= 0.]))
        ind_stim_i = np.mean(ind_stim[:, 0])
        ind_stim_j = np.mean(ind_stim[:, 1])
        ind_stim_k = np.mean(ind_stim[:, 2])

        return euclidean([ind_stim_i, ind_stim_j, ind_stim_k], [ind_front_i, ind_front_j, ind_front_k])*dr/max_act

    @property
    def output(self):
        return self.compute_velocity_front()

    def write(self):
        jdata = json.dumps(self.compute_velocity_front())
        with open(self.file_name, "w") as jf:
            jf.write(jdata)
