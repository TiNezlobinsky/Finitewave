from typing import Any, Callable


class KernelSpec:
    """Backend-independent kernel specs."""

    name: str
    ops: Any
    step_function: Callable
    arrays: tuple[str, ...]
    scalars: tuple[str, ...]
    state_vars: tuple[str, ...]
    observers: tuple[Any, ...]
    output_args: tuple[str, ...]
