import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt

from finitewave.cpuwave3D.model.aliev_panfilov_3d import AlievPanfilov3D
from finitewave.cpuwave3D.tissue.cardiac_tissue_3d import CardiacTissue3D
from finitewave.cpuwave3D.tracker.period_3d_tracker import Period3DTracker
from finitewave.cpuwave3D.tracker.velocity_3d_tracker import Velocity3DTracker
from finitewave.cpuwave3D.stimulation.stim_current_coord_3d import StimCurrentCoord3D
from finitewave.cpuwave3D.stimulation.stim_voltage_coord_3d import StimVoltageCoord3D
from finitewave.cpuwave3D.stencil.asymmetric_stencil_3d import AsymmetricStencil3D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence


class TestAlievPanfilov3D(unittest.TestCase):
    def setUp(self):

        n = 100
        self.tissue = CardiacTissue3D([n, n, n], mode='aniso')
        self.tissue.mesh = np.ones([n, n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, n, 3])
        self.tissue.stencil = AsymmetricStencil3D()

        self.aliev_panfilov = AlievPanfilov3D()
        self.aliev_panfilov.dt    = 0.01
        self.aliev_panfilov.dr    = 0.25
        self.aliev_panfilov.t_max = 40

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimCurrentCoord3D(0, 3, 0.18, 0, 100, 0, 5, 0, 5))

        tracker_sequence = TrackerSequence()
        self.velocity_tracker = Velocity3DTracker()
        self.velocity_tracker.threshold = 0.2
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.aliev_panfilov.cardiac_tissue   = self.tissue
        self.aliev_panfilov.stim_sequence  = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,:,0] = 0.
        self.tissue.fibers[:,:,:,1] = 1.
        self.tissue.fibers[:,:,:,2] = 0.
        self.tissue.D_al  = 1.
        self.tissue.D_ac = 0.111

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimCurrentCoord3D(0, 3, 0.18, 0, 100, 0, 5, 0, 100))

        self.aliev_panfilov.stim_sequence  = stim_sequence

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 1.6,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 3D)",
                               delta=0.05)

    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,:,0] = 1.
        self.tissue.fibers[:,:,:,1] = 0.
        self.tissue.fibers[:,:,:,2] = 0.
        self.tissue.D_al  = 1.
        self.tissue.D_ac = 0.111

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.5,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 3D)",
                               delta=0.05)
