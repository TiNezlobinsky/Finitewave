from types import SimpleNamespace

import numpy as np

import finitewave as fw


class IdentityBackend:
    @staticmethod
    def wrap_indexes(indexes):
        return indexes


def test_legacy_voltage_coord_matches_grid_box():
    tissue = fw.CardiacTissue(shape=(4, 5), dr=0.25)
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        backend=IdentityBackend(),
    )
    legacy = fw.StimVoltageCoord(0, 1, 1, 3, 2, 5)
    modern = fw.StimVoltageGridBox(0, 1, 1, 3, 2, 5)

    legacy.initialize(simulation)
    modern.initialize(simulation)

    np.testing.assert_array_equal(legacy.stim_indexes, modern.stim_indexes)


def test_legacy_current_coord_matches_grid_box():
    tissue = fw.CardiacTissue(shape=(4, 5), dr=0.25)
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        backend=IdentityBackend(),
    )
    legacy = fw.StimCurrentCoord(0, 2, 0.5, 1, 3, 2, 5)
    modern = fw.StimCurrentGridBox(0, 2, 0.5, 1, 3, 2, 5)

    legacy.initialize(simulation)
    modern.initialize(simulation)

    np.testing.assert_array_equal(legacy.stim_indexes, modern.stim_indexes)


def test_legacy_voltage_matrix_matches_voltage_mask():
    tissue = fw.CardiacTissue(shape=(3, 3), dr=0.25)
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        backend=IdentityBackend(),
    )
    mask = np.array(
        [[True, False, False],
         [False, True, False],
         [False, False, True]]
    )
    legacy = fw.StimVoltageMatrix(0, 1, mask)
    modern = fw.StimVoltageMask(0, 1, mask)

    legacy.initialize(simulation)
    modern.initialize(simulation)

    np.testing.assert_array_equal(legacy.stim_indexes, modern.stim_indexes)
