import numpy as np
import pytest

import finitewave as fw


def test_single_cell_simulation_creates_stimulation():
    model = fw.AlievPanfilov()

    simulation = fw.SingleCellSimulation(model=model, dt=0.1)

    assert isinstance(simulation.stimulation, fw.SingleCellStimulation)
    assert simulation.stimulation.dt == 0.1


def test_single_cell_stimulation_builds_consecutive_protocol_segments():
    stimulation = fw.SingleCellStimulation(dt=0.5)
    stimulation.add_stimulation(
        n_beats=2,
        cycle_length=2.0,
        current=3.0,
        duration=0.5,
    )
    stimulation.add_stimulation(
        n_beats=1,
        cycle_length=1.0,
        current=2.0,
        duration=0.5,
    )

    np.testing.assert_array_equal(
        stimulation.stim_current,
        [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    )


def test_single_cell_stimulation_accepts_explicit_current_trace():
    stimulation = fw.SingleCellStimulation(dt=0.1)

    stimulation.stim_current = [1.0, 0.5, 0.0]

    np.testing.assert_array_equal(stimulation.stim_current, [1.0, 0.5, 0.0])


def test_single_cell_simulation_rejects_mismatched_time_steps():
    stimulation = fw.SingleCellStimulation(dt=0.2)

    with pytest.raises(ValueError, match="same dt"):
        fw.SingleCellSimulation(
            model=fw.AlievPanfilov(),
            dt=0.1,
            stimulation=stimulation,
        )
