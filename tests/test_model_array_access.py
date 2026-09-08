import copy

import numpy as np
import pytest

import finitewave as fw
from finitewave.core.stimul.stim_type.stim_voltage import StimVoltage


@pytest.mark.parametrize("backend_name", ["numba", "jax"])
def test_public_arrays_and_backend_simulation(backend_name):
    if backend_name == "jax":
        pytest.importorskip("jax")
    model = fw.AlievPanfilov()
    assert model._u is None
    with pytest.raises(AttributeError):
        model.u
    parameter = model.state_pars[0]
    model.set_parameters({parameter: np.full(16, getattr(model, parameter))})
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02, backend=backend_name)
    tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_tissue = tissue
    simulation.cardiac_model = model
    simulation.initialize()
    assert isinstance(getattr(model, parameter), np.ndarray)
    np.testing.assert_array_equal(
        getattr(model, parameter).ravel(), np.asarray(getattr(model, f"_{parameter}"))
    )
    stimulus = StimVoltage(time=0, volt_value=0.75)
    stimulus.stim_indexes = simulation.backend.wrap_indexes(np.array([0, 1]))
    stimulus.stimulate(simulation)
    np.testing.assert_allclose(model.u.ravel()[:2], 0.75)

    for name in model.state_vars:
        public = getattr(model, name)
        wrapped = getattr(model, f"_{name}")
        assert isinstance(public, np.ndarray)
        np.testing.assert_array_equal(public.ravel(), np.asarray(wrapped))
        if backend_name == "jax":
            import jax
            assert isinstance(wrapped, jax.Array)
    assert model.u.shape == (4, 4)
    assert model.output("u").shape == (4, 4)

    # Explicit updates must reach the cached kernel inputs.
    other = next(name for name in model.state_vars if name != "u")
    model.set_values({other: np.full(16, 0.1)})
    assert model.model_kernel_args[model.kernel_arg_names.index(other)] is getattr(model, f"_{other}")
    model.update_state_variables({"u": 0.5, other: 0.2})
    np.testing.assert_allclose(model.u, 0.5)
    np.testing.assert_allclose(getattr(model, other), 0.2)

    simulation.run(initialize=False, prog_bar=False)
    assert np.all(np.isfinite(model.u))
    assert not np.allclose(model.u, 0.5)
    if backend_name == "jax":
        assert isinstance(model._u, jax.Array)
        assert isinstance(model._rhs, jax.Array)
    for name in model.state_vars:
        np.testing.assert_array_equal(getattr(model, name).ravel(), np.asarray(getattr(model, f"_{name}")))
    cloned = copy.copy(model)
    np.testing.assert_array_equal(cloned.u, model.u)
    with pytest.raises(AttributeError):
        model.missing_variable


def test_output_expands_compact_numpy_array():
    model = fw.AlievPanfilov()
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02)
    simulation.cardiac_tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_tissue.mesh[0, 0] = 0
    model.initialize(simulation)
    assert model.u.shape == (15,)
    assert model.output("u").shape == (4, 4)
    assert np.isnan(model.output("u")[0, 0])


@pytest.mark.parametrize("backend_name", ["numba", "jax"])
def test_model_values_require_explicit_updates(backend_name):
    if backend_name == "jax":
        pytest.importorskip("jax")
    model = fw.AlievPanfilov()
    with pytest.raises(RuntimeError, match="Initialize"):
        model.set_values({"u": 1.})
    with pytest.raises(AttributeError, match="Cannot assign"):
        model.u = 1.
    model.init_u = 0.25
    simulation = fw.CardiacSimulation(dt=0.01, t_max=0.02, backend=backend_name)
    simulation.cardiac_tissue = fw.CardiacTissueGrid((4, 4), dr=0.25)
    simulation.cardiac_model = model
    simulation.initialize()
    parameter = model.state_pars[0]
    field = np.full((4, 4), getattr(model, parameter))
    model.set_parameters({parameter: field})
    field[:] = -123
    assert not np.any(getattr(model, parameter) == -123)
    for name in list(model.state_vars) + [parameter, "rhs"]:
        before = np.array(getattr(model, f"_{name}"), copy=True)
        with pytest.raises(AttributeError, match="Cannot assign"):
            setattr(model, name, np.ones((4, 4)))
        with pytest.raises(ValueError, match="read-only"):
            getattr(model, name)[:] = 1.
        snapshot = getattr(model, name)
        snapshot.setflags(write=True)
        snapshot[:] = -123
        np.testing.assert_array_equal(getattr(model, f"_{name}"), before)
    values = np.full(16, 0.5)
    model.set_values({"u": values})
    values[:] = -123
    np.testing.assert_allclose(model.u, 0.5)
    with pytest.raises(ValueError):
        model.set_values({"u": 0.8, parameter: np.zeros(3)})
    np.testing.assert_allclose(model.u, 0.5)
    simulation.run(initialize=False, prog_bar=False)
    assert np.all(np.isfinite(model.u))
