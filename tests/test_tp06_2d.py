import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys

from finitewave.cpuwave2D.model.tp06_2d import TP062D
from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.tracker.velocity_2d_tracker import Velocity2DTracker
from finitewave.cpuwave2D.tracker.period_2d_tracker import Period2DTracker
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import AsymmetricStencil2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence


class TestTP062D(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = CardiacTissue2D([n, n], mode='aniso')
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])
        self.tissue.stencil = AsymmetricStencil2D()
        self.tissue.D_al  = 0.154
        self.tissue.D_ac  = 0.154

        self.tp06 = TP062D()
        self.tp06.dt    = 0.001
        self.tp06.dr    = 0.1
        self.tp06.t_max = 10

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, 200, 0, 5))

        tracker_sequence = TrackerSequence()
        self.velocity_tracker = Velocity2DTracker()
        self.velocity_tracker.threshold = -60
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.tp06.cardiac_tissue   = self.tissue
        self.tp06.stim_sequence    = stim_sequence
        self.tp06.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, 200, 0, 5))
        self.tp06.stim_sequence  = stim_sequence

        self.tp06.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())

        self.assertAlmostEqual(front_vel, 0.8,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)


    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,0] = 1.
        self.tissue.fibers[:,:,1] = 0.
        self.tissue.D_al = 0.154
        self.tissue.D_ac = 0.0171

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimVoltageCoord2D(0, -20, 0, 200, 0, 5))
        self.tp06.stim_sequence  = stim_sequence

        self.tp06.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.23,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    # def test_spiral_wave_period(self):
    #     sys.stdout.write("---> Check the spiral wave period\n")
    #    self.tissue.fibers[:,:,0] = 1.
    #    self.tissue.fibers[:,:,1] = 0.
    #    self.tissue.D_al = 0.154
    #    self.tissue.D_ac = 0.154
    #
    #     stim_params = [[0, 200, 0, 100, -20, 0.],
    #                    [0, 100, 0, 200, -20, 31.]]
    #     self.tp06.cardiac_tissue = self.tissue
    #     self.tp06.stim_params    = stim_params
    #
    #     period_tracker = Period2DTracker()
    #     period_tracker.target_model = self.tp06
    #     period_tracker.mode = "Detectors"
    #
    #     detectors = np.zeros([200, 200], dtype="uint8")
    #     positions = np.array([[100, 100]])
    #     detectors[positions[:, 0], positions[:, 1]] = 1
    #
    #     period_tracker.detectors = detectors
    #     period_tracker.threshold = 0.2
    #
    #     # add tracker to the model
    #     self.tp06.add_tracker(period_tracker)
    #
    #     self.tp06.run()
    #
    #     self.assertAlmostEqual(period_tracker.output["100,100"][-1][1], 194.,
    #                            msg="Spiral wave period is incorrect! (AlievPanfilov 2D)",
    #                            delta=1.)
