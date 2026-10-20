import numpy as np

from .spec import KernelSpec


class ModelAnalyzer:
    """Collect model metadata without selecting an execution backend."""

    def analyze(self, model):
        arrays = list(model.state_vars)
        scalars = []

        for name in model.state_pars:
            value = getattr(model, name)
            if np.isscalar(value):
                scalars.append(name)
            else:
                arrays.append(name)

        return KernelSpec(
            name="ionic_kernel",
            ops=model.ops,
            step_function=model.ops.ionic_step,
            arrays=tuple(arrays),
            scalars=tuple(scalars),
            state_vars=tuple(model.state_vars),
            observers=tuple(model.observers),
            output_args=(
                "rhs",
                *(name for name in model.state_vars if name != "u"),
            ),
        )
