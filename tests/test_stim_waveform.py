from types import SimpleNamespace

import numpy as np

import finitewave as fw


class NumpyBackend:
    @staticmethod
    def wrap_indexes(indexes):
        return indexes

    @staticmethod
    def set_flat_values(array, indexes, value):
        array[indexes] = value
        return array


def test_voltage_waveform_applies_one_sample_per_time_step():
    tissue = fw.CardiacTissue(shape=(3, 1), dr=1.0)
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        cardiac_model=SimpleNamespace(_u=np.zeros(3)),
        backend=NumpyBackend(),
        dt=0.1,
        t=0.0,
    )
    spatial_stimulus = fw.StimVoltageGridBox(
        time=0.0,
        voltage=0.0,
        x_min=1,
        x_max=2,
        y_min=0,
        y_max=1,
    )
    waveform = fw.StimVoltageWaveform(
        spatial_stimulus,
        values=[0.2, 0.5, 0.9],
    )
    waveform.initialize(simulation)

    observed = []
    for time in [0.0, 0.1, 0.2]:
        simulation.t = time
        waveform.stimulate(simulation)
        observed.append(simulation.cardiac_model._u.copy())

    np.testing.assert_allclose(
        observed,
        [[0.0, 0.2, 0.0], [0.0, 0.5, 0.0], [0.0, 0.9, 0.0]],
    )
    assert waveform.duration == 0.2
