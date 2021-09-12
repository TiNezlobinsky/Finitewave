import numpy as np
import sys
import tqdm
import math
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.gpuwave2D.model.cuda_sources import Kernels


class TP062D(CardiacModel):
    def __init__(self):
        CardiacModel.__init__(self)

        self.mode = """"""

        self.prog_bar = True
        self.npfloat  = "float32"
        self.threshold = np.float32(-40)
        self.Di = 0.154
        self.Dj = 0.154

        self.act_t   = np.array([])

        self.Cai     = np.array([])
        self.CaSR    = np.array([])
        self.CaSS    = np.array([])
        self.Nai     = np.array([])
        self.Ki      = np.array([])
        self.M_      = np.array([])
        self.H_      = np.array([])
        self.J_      = np.array([])
        self.Xr1     = np.array([])
        self.Xr2     = np.array([])
        self.Xs      = np.array([])
        self.R_      = np.array([])
        self.S_      = np.array([])
        self.D_      = np.array([])
        self.F_      = np.array([])
        self.F2_     = np.array([])
        self.FCass   = np.array([])
        self.RR      = np.array([])
        self.OO      = np.array([])
        self.weights = np.array([])

        self.u_new_d = None
        self.u_d     = None
        self.c_d     = None
        self.w_d     = None
        self.act_t_d = None
        self.stim_mask_d = None # for stimulation

        self.Cai_d   = None
        self.CaSR_d  = None
        self.CaSS_d  = None
        self.Nai_d   = None
        self.Ki_d    = None
        self.M__d     = None
        self.H__d     = None
        self.J__d     = None
        self.Xr1_d   = None
        self.Xr2_d   = None
        self.Xs_d    = None
        self.R__d     = None
        self.S__d     = None
        self.D__d     = None
        self.F__d     = None
        self.F2__d    = None
        self.FCass_d = None
        self.RR_d    = None
        self.OO_d    = None

        self.state_vars = ["u", "Cai", "CaSR", "CaSS", "Nai", "Ki", "M_", "H_", "J_", "Xr1", "Xr2", "Xs", "R_", "S_",
                                    "D_", "F_", "F2_", "FCass", "RR", "OO"]

    def _initialization(self):

        self.size_i = self.cardiac_tissue.size_i
        self.size_j = self.cardiac_tissue.size_j

        self.block_size = (32, 8, 1) # !blocksize affects to shared memory
        grid_x = int(math.ceil(self.size_j / self.block_size[0]))
        grid_y = int(math.ceil(self.size_i / self.block_size[1]))
        self.grid_size = (grid_x, grid_y, 1)

        self._vars = {"Cai":0.00007, "CaSR":1.3, "CaSS":0.00007,
                      "Nai":7.67, "Ki":138.3, "M_":0., "H_":0.75,
                      "J_":0.75, "Xr1":0., "Xr2":1., "Xs":0., "R_":0.,
                      "S_":1., "D_":0., "F_":1., "F2_":1., "FCass":1.,
                      "RR":1., "OO":0.}

        self.u = -84.5 * np.ones([self.size_i, self.size_j], dtype=self.npfloat)
        self.cond    = self.cardiac_tissue.cond.astype(self.npfloat)
        self.weights = self.cardiac_tissue.compute_weights(self.Di, self.Dj).astype(self.npfloat)
        self.act_t   = -np.ones([self.size_i, self.size_j], dtype=self.npfloat)

        for vars, vals in self._vars.items():
            self.__dict__[vars] = vals * np.ones([self.size_i, self.size_j],
                                                 dtype=self.npfloat)

        self.mod = SourceModule(Kernels.get_stim_curr_kernel() +
                                Kernels.get_stim_volt_kernel() +
                                Kernels.get_act_time_kernel() +
                                Kernels.get_diff_kernel() +
                                Kernels.get_tp06_curr_kernel())
        # add ..., options=['-use_fast_math']) to gain performance but lose
        # precision

        self.stim_curr_kernel = self.mod.get_function("stim_curr")
        self.stim_volt_kernel = self.mod.get_function("stim_volt")
        self.act_time_kernel  = self.mod.get_function("act_time")
        self.diffusion_kernel = self.mod.get_function("diffusion")
        self.currents_kernel  = self.mod.get_function("currents")

        if self.stim_sequence:
            self.stim_sequence.initialize(self)
        if self.tracker_sequence:
            self.tracker_sequence.initialize(self)
        if self.state_keeper and self.state_keeper.record_load:
            self.state_keeper.load(self)

    def run(self):

        if self.prog_bar:
            max_step = self.t_max/self.dt
            self.bar = tqdm.tqdm(total=max_step) # max = 100%

        # manual cuda init for compability with numba cuda
        cuda.init()
        current_dev = cuda.Device(0)
        ctx = current_dev.make_context()
        ctx.push()

        self.t    = 0
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
        for var, value in self._vars.items():
            self.get_array(var)

    def get_array(self, target_array):
        cuda.memcpy_dtoh(self.__dict__[target_array], self.__dict__[target_array+"_d"])

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
        self.t  = np.float32(self.t)

    def _data_to_device(self):
        self.u_new_d = cuda.to_device(self.u)
        self.u_d     = cuda.to_device(self.u)
        self.c_d     = cuda.to_device(self.cond)
        self.act_t_d = cuda.to_device(self.act_t)
        self.stim_mask_d = cuda.to_device(np.ones([self.size_i, self.size_j],
                                                  dtype=self.npfloat))
        self.w_d = [None]*self.weights.shape[2]
        for i in range(0, self.weights.shape[2]):
            if i not in [1, 4]:
                self.w_d[i] = cuda.to_device(self.weights[:, :, i])
        # to prevent additional acces to global memory
        self.w_d[1] = cuda.to_device(self.weights[:, :, 1] +
                                     self.weights[:, :, 4])

        for vars in self._vars:
            self.__dict__[vars+"_d"] = cuda.to_device(self.__dict__[vars])

    def _run_kernel(self):

        t_max = self.t_max - self.dt/2

        while self.t < t_max:
            if self.stim_sequence:
                self.stim_sequence.stimulate_next()

            self.diffusion_kernel(self.u_new_d, self.u_d, self.c_d,
                      self.w_d[0], self.w_d[1], self.w_d[2], self.w_d[3],
                      self.w_d[5], self.w_d[6], self.w_d[7], self.dt, self.dr,
                      self.size_i, self.size_j,
                      block=self.block_size, grid=self.grid_size)

            self.currents_kernel(self.u_new_d, self.u_d, self.Cai_d, self.CaSR_d,
                     self.CaSS_d, self.Nai_d, self.Ki_d, self.M__d, self.H__d,
                     self.J__d, self.Xr1_d, self.Xr2_d, self.Xs_d, self.R__d,
                     self.S__d, self.D__d, self.F__d, self.F2__d, self.FCass_d,
                     self.RR_d, self.OO_d, self.dt, self.size_i, self.size_j,
                     block=self.block_size, grid=self.grid_size)

            self.u_d, self.u_new_d = self.u_new_d, self.u_d

            if self.tracker_sequence:
                self.tracker_sequence.tracker_next()

            self.step += 1
            self.t     = np.float32(self.step*self.dt)

            if self.prog_bar:
                self.bar.update(1)

        if self.prog_bar:
            self.bar.close()

        self.u[self.cardiac_tissue.mesh != 1] = -86.

        if self.state_keeper and self.state_keeper.record_save:
            cuda.memcpy_dtoh(self.u, self.u_d)
            for vars in self._vars:
                cuda.memcpy_dtoh(self.__dict__[vars], self.__dict__[vars+'_d'])
            self.state_keeper.save(self)
