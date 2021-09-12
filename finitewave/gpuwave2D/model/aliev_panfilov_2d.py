import numpy as np
import sys
import tqdm
import math
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from copy import deepcopy

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.gpuwave2D.model.cuda_sources import Kernels


class AlievPanfilov2D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)

        self.mod = None

        self.prog_bar = True
        self.threshold = np.float32(-40)
        self.npfloat = "float32"  # float32 only

        self.a = np.float32(0.1)
        self.k = np.float32(8.)
        self.eap = np.float32(0.01)
        self.mu_1 = np.float32(0.2)
        self.mu_2 = np.float32(0.3)

        self.u_new_d = None
        self.u_d = None
        self.v_d = None
        self.c_d = None
        self.w_d = None
        self.act_t_d = None
        self.stim_mask_d = None  # for stimulation

        self.v = np.array([])
        self.weights = np.array([])
        self.act_t = np.array([])

        self.state_vars = ["u", "v"]

    def _initialization(self):
        self.size_i = self.cardiac_tissue.size_i
        self.size_j = self.cardiac_tissue.size_j

        self.block_size = (32, 8, 1)  # !blocksize affects to shared memory
        grid_x = int(math.ceil(self.size_j / self.block_size[0]))
        grid_y = int(math.ceil(self.size_i / self.block_size[1]))
        self.grid_size = (grid_x, grid_y, 1)

        self.u = np.zeros([self.size_i, self.size_j], dtype=self.npfloat)
        self.v = np.zeros([self.size_i, self.size_j], dtype=self.npfloat)
        self.cond = self.cardiac_tissue.cond.astype(self.npfloat)
        self.weights = self.cardiac_tissue.compute_weights(
            self.Di, self.Dj).astype(self.npfloat)
        self.act_t = -np.ones([self.size_i, self.size_j], dtype=self.npfloat)

        self.mod = SourceModule(Kernels.get_stim_curr_kernel() +
                                Kernels.get_stim_volt_kernel() +
                                Kernels.get_act_time_kernel() +
                                Kernels.get_diff_kernel() +
                                Kernels.get_ap_curr_kernel())
        # add ..., options=['-use_fast_math']) to gain performance but lose
        # precision

        self.stim_curr_kernel = self.mod.get_function("stim_curr")
        self.stim_volt_kernel = self.mod.get_function("stim_volt")
        self.act_time_kernel = self.mod.get_function("act_time")
        self.diffusion_kernel = self.mod.get_function("diffusion")
        self.currents_kernel = self.mod.get_function("currents")

        if self.stim_sequence:
            self.stim_sequence.initialize(self)
        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)
        if self.state_keeper and self.state_keeper.record_load:
            self.state_keeper.load(self)

    def run(self):

        if self.prog_bar:
            max_step = self.t_max/self.dt
            self.bar = tqdm.tqdm(total=max_step)  # max = 100%
        # manual cuda init for compability with numba cuda
        cuda.init()
        current_dev = cuda.Device(0)
        ctx = current_dev.make_context()
        ctx.push()

        self.t = 0
        self.step = 0
        self._initialization()
        self._params_to_float32()

        try:
            self._data_to_device()
            self._run_kernel()
        except:
            ctx.pop()
            ctx.detach()
            raise
        ctx.pop()
        ctx.detach()

    def load_state(self):
        self.get_array('u')
        self.get_array('v')

    def get_array(self, target_array):
        cuda.memcpy_dtoh(self.__dict__[target_array],
                         self.__dict__[target_array+"_d"])

    def stim_curr(self, stim_mask, value):
        cuda.memcpy_htod(self.stim_mask_d, stim_mask)
        self.stim_curr_kernel(self.u_d, self.stim_mask_d, value, self.size_i,
                              self.size_j, block=self.block_size,
                              grid=self.grid_size)

    def stim_volt(self, stim_mask, value):
        cuda.memcpy_htod(self.stim_mask_d, stim_mask)
        self.stim_volt_kernel(self.u_d, self.stim_mask_d, value, self.size_i,
                              self.size_j, block=self.block_size,
                              grid=self.grid_size)

    def track_act_time(self):
        self.act_time_kernel(self.act_t_d, self.u_d, self.threshold, self.t,
                             self.size_i, self.size_j, block=self.block_size,
                             grid=self.grid_size)

    def _params_to_float32(self):
        self.size_i = np.int32(self.size_i)
        self.size_j = np.int32(self.size_j)
        self.dt = np.float32(self.dt)
        self.dr = np.float32(self.dr)
        self.t = np.float32(self.t)

    def _data_to_device(self):
        self.u_new_d = cuda.to_device(self.u)
        self.u_d = cuda.to_device(self.u)
        self.v_d = cuda.to_device(self.v)
        self.c_d = cuda.to_device(self.cond)
        self.act_t_d = cuda.to_device(self.act_t)
        self.stim_mask_d = cuda.to_device(np.ones(shape=self.u.shape,
                                                  dtype=self.npfloat))
        self.w_d = [None]*self.weights.shape[2]
        for i in range(0, self.weights.shape[2]):
            if i not in [1, 4]:
                self.w_d[i] = cuda.to_device(self.weights[:, :, i])
        # to prevent additional acces to global memory
        self.w_d[1] = cuda.to_device(self.weights[:, :, 1] +
                                     self.weights[:, :, 4])

    def _run_kernel(self):

        millis = 0

        t_max = self.t_max - self.dt/2

        while self.t < t_max:
            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            self.diffusion_kernel(self.u_new_d, self.u_d, self.c_d,
                                  self.w_d[0], self.w_d[1], self.w_d[2], self.w_d[3],
                                  self.w_d[5], self.w_d[6], self.w_d[7], self.dt, self.dr,
                                  self.size_i, self.size_j,
                                  block=self.block_size, grid=self.grid_size)

            self.currents_kernel(self.u_new_d, self.u_d, self.v_d, self.dt,
                                 self.a, self.k, self.eap, self.mu_1, self.mu_2,
                                 self.size_i, self.size_j,
                                 block=self.block_size, grid=self.grid_size)

            self.u_d, self.u_new_d = self.u_new_d, self.u_d

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.step += 1
            self.t = np.float32(self.step * self.dt)

            if self.prog_bar:
                self.bar.update(1)

        if self.prog_bar:
            self.bar.close()

        self.u[self.cardiac_tissue.mesh != 1] = 0.

        if self.state_keeper and self.state_keeper.record_save:
            cuda.memcpy_dtoh(self.u, self.u_d)
            cuda.memcpy_dtoh(self.v, self.v_d)
            self.state_keeper.save(self)
