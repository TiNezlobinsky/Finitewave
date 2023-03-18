import unittest
import numpy as np

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.model.aliev_panfilov_2d import AlievPanfilov2D
from finitewave.cpuwave2D.stimulation.stim_voltage_coord_2d import StimVoltageCoord2D

from finitewave.core.stimulation.stim_sequence import StimSequence
from finitewave.core.tracker.tracker_sequence import TrackerSequence

from finitewave.cpuwave2D.tracker.multi_activation_time_2d_tracker import MultiActivationTime2DTracker
from finitewave.cpuwave2D.tracker.multivariable_2d_tracker import MultiVariable2DTracker


def extract_activation_times(t, u, thr):
    activation_times = []
    activated = False
    for i in range(len(t)):
        if u[i] > thr and not activated:
            activation_times.append(t[i])
            activated = True
        elif u[i] <= thr and activated:
            activated = False
    return activation_times


class TestMultiActTime(unittest.TestCase):
    def setUp(self):

        n = 200
        self.tissue = CardiacTissue2D([n, n], mode='aniso')
        self.tissue.mesh = np.ones([n, n], dtype="uint8")
        self.tissue.add_boundaries()
        self.tissue.fibers = np.zeros([n, n, 2])

        self.aliev_panfilov = AlievPanfilov2D()
        self.aliev_panfilov.dt    = 0.01
        self.aliev_panfilov.dr    = 0.25
        self.aliev_panfilov.t_max = 25

        stim_sequence = StimSequence()
        stim_sequence.add_stim(StimVoltageCoord2D(0, 1, 0, 3, 0, n))
        stim_sequence.add_stim(StimVoltageCoord2D(100, 1, 0, 3, 0, n))
        stim_sequence.add_stim(StimVoltageCoord2D(200, 1, 0, 3, 0, n))

        tracker_sequence = TrackerSequence()

        self.act_time_tracker = MultiActivationTime2DTracker()
        self.act_time_tracker.threshold = 0.5
        tracker_sequence.add_tracker(self.act_time_tracker)

        self.multivariable_tracker = MultiVariable2DTracker()
        self.multivariable_tracker.cell_ind = [100, 100]
        self.multivariable_tracker.var_list = ["u"]
        tracker_sequence.add_tracker(self.multivariable_tracker)

        tracker_sequence.add_tracker(self.act_time_tracker)
        tracker_sequence.add_tracker(self.multivariable_tracker)

        self.aliev_panfilov.cardiac_tissue   = self.tissue
        self.aliev_panfilov.stim_sequence    = stim_sequence
        self.aliev_panfilov.tracker_sequence = tracker_sequence

    def test_activation_time_sequence(self):
        self.aliev_panfilov.run()

        calculated_times = []
        for i in range(len(self.act_time_tracker.act_t)):
            calculated_times.append(self.act_time_tracker.act_t[i][100][100])

        reference_times = extract_activation_times(np.arange(len(self.multivariable_tracker.vars["u"]))*self.aliev_panfilov.dt,
                                                   self.multivariable_tracker.vars["u"],
                                                   0.5)
        self.assertEqual(len(calculated_times), len(reference_times),
                         msg="Activation time sequence has incorrect length (Multi activation time)")

        for i in range(len(calculated_times)):
            self.assertAlmostEqual(calculated_times[i], reference_times[i],
                                   msg="Different activation times sequence (Multi activation time)",
                                   delta=2*self.aliev_panfilov.dt)
