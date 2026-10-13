from types import SimpleNamespace

import numpy as np

import finitewave as fw


def test_physical_box_uses_half_open_bounds():
    box = fw.PhysicalBox(lower=(0, 0), upper=(2, 2))
    points = np.array([[0, 0], [1, 1], [2, 1], [1, 2]])

    np.testing.assert_array_equal(
        box.contains(points),
        [True, True, False, False],
    )


def test_fdm_slice_selection_returns_compact_state_indexes():
    mesh = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.int8)
    tissue = fw.CardiacTissue(mesh=mesh, dr=0.5)
    selection = fw.GridSliceSelection(slice(0, 2), slice(1, 3))

    np.testing.assert_array_equal(selection.select(tissue), [1, 3, 4])


def test_fdm_physical_selection_accounts_for_grid_spacing():
    tissue = fw.CardiacTissue(shape=(4, 2), dr=0.5)
    region = fw.PhysicalBox(lower=(0.5, 0.0), upper=(1.5, 1.0))

    indexes = fw.GridRegionSelection(region).select(tissue)

    np.testing.assert_array_equal(indexes, [2, 3, 4, 5])


def test_fem_region_selection_uses_physical_node_coordinates():
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    tissue = fw.CardiacTissueElements(coords, elems, fw.ElementType.TRIANGLE)
    region = fw.PhysicalBox(lower=(0.0, 0.0), upper=(0.5, 1.1))

    indexes = fw.NodeRegionSelection(region).select(tissue)

    np.testing.assert_array_equal(indexes, [0, 2])


def test_stimulus_wraps_indexes_returned_by_selection():
    class RecordingBackend:
        def __init__(self):
            self.received = None

        def wrap_indexes(self, indexes):
            self.received = indexes
            return ("wrapped", *indexes)

    tissue = fw.CardiacTissue(shape=(3, 2), dr=1.0)
    backend = RecordingBackend()
    simulation = SimpleNamespace(cardiac_tissue=tissue, backend=backend)
    selection = fw.GridSliceSelection(slice(1, 3), slice(0, 1))
    stimulus = fw.StimVoltage(0, 1, selection=selection)

    stimulus.initialize(simulation)

    np.testing.assert_array_equal(backend.received, [2, 4])
    assert stimulus.stim_indexes == ("wrapped", 2, 4)


def test_physical_and_grid_boxes_have_explicitly_different_units():
    class IdentityBackend:
        @staticmethod
        def wrap_indexes(indexes):
            return indexes

    tissue = fw.CardiacTissue(shape=(6, 1), dr=0.5)
    simulation = SimpleNamespace(
        cardiac_tissue=tissue,
        backend=IdentityBackend(),
    )
    grid_stimulus = fw.StimVoltageGridBox(
        time=0, voltage=1,
        x_min=0, x_max=2, y_min=0, y_max=1,
    )
    physical_stimulus = fw.StimVoltageBox(
        time=0, voltage=1,
        x_min=0, x_max=2, y_min=0, y_max=1,
    )

    grid_stimulus.initialize(simulation)
    physical_stimulus.initialize(simulation)

    np.testing.assert_array_equal(grid_stimulus.stim_indexes, [0, 1])
    np.testing.assert_array_equal(
        physical_stimulus.stim_indexes, [0, 1, 2, 3]
    )
