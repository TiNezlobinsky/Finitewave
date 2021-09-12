import sys
import unittest
import numpy as np
import matplotlib.pyplot as plt

from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.tracker.velocity_2d_tracker import Velocity2DTracker
from finitewave.cpuwave2D.fibrosis.diffuse_2d_pattern import Diffuse2DPattern
from finitewave.cpuwave2D.tracker.period_2d_tracker import Period2DTracker
from finitewave.cpuwave2D.tracker.spiral_2d_tracker import Spiral2DTracker
from finitewave.cpuwave2D.stimulation.stim_current_coord_2d import StimCurrentCoord2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import AsymmetricStencil2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence


class TestAlievPanfilov2D(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = CardiacTissue2D([n, n], mode='aniso')
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])
        self.tissue.stencil = AsymmetricStencil2D()

        self.aliev_panfilov = AlievPanfilov2D()
        self.aliev_panfilov.dt    = 0.01
        self.aliev_panfilov.dr    = 0.25
        self.aliev_panfilov.t_max = 25

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimCurrentCoord2D(0, 3, 0.18, 0, 200, 0, 5))

        tracker_sequence = TrackerSequence()
        self.velocity_tracker = Velocity2DTracker()
        self.velocity_tracker.threshold = 0.2
        tracker_sequence.add_tracker(self.velocity_tracker)

        self.aliev_panfilov.cardiac_tissue   = self.tissue
        self.aliev_panfilov.stim_sequence    = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

    def test_wave_along_the_fibers(self):
        sys.stdout.write("---> Check the wave speed along the fibers\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 1.6,
                               msg="Wave velocity along the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    def test_wave_across_the_fibers(self):
        sys.stdout.write("---> Check the wave speed across the fibers\n")
        self.tissue.fibers[:,:,0] = 1.
        self.tissue.fibers[:,:,1] = 0.
        self.tissue.D_al = 1
        self.tissue.D_ac = 0.111

        self.aliev_panfilov.run()

        front_vel = np.mean(self.velocity_tracker.compute_velocity_front())
        self.assertAlmostEqual(front_vel, 0.55,
                               msg="Wave velocity across the fibers direction is incorrect! (AlievPanfilov 2D)",
                               delta=0.05)

    def test_spiral_wave_period(self):
        sys.stdout.write("---> Check the spiral wave period\n")
        self.tissue.fibers[:,:,0] = 0.
        self.tissue.fibers[:,:,1] = 1.
        self.tissue.D_al = 1.
        self.tissue.D_ac = 1.
        self.aliev_panfilov.t_max = 200

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, 200, 0, 100))
        stim_sequence.add_stim(StimVoltageCoord2D(31, 1, 0, 100, 0, 200))

        tracker_sequence = TrackerSequence()
        period_tracker = Period2DTracker()
        detectors = np.zeros([200, 200], dtype="uint8")
        positions = np.array([[100, 100]])
        detectors[positions[:, 0], positions[:, 1]] = 1
        period_tracker.detectors = detectors
        period_tracker.threshold = 0.2
        tracker_sequence.add_tracker(period_tracker)

        spiral_tracker = Spiral2DTracker()
        tracker_sequence.add_tracker(spiral_tracker)

        self.aliev_panfilov.stim_sequence    = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

        self.aliev_panfilov.run()

        self.assertAlmostEqual(period_tracker.output["100,100"][-1][1], 25.8,
                               msg="Spiral wave period is incorrect! (AlievPanfilov 2D)",
                               delta=0.3)

        spiral_tracker.write()
