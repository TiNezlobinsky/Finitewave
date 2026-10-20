from dataclasses import dataclass
from typing import Callable

from .spec import KernelSpec


@dataclass
class KernelArtifact:
    """Generated forms of a kernel from source through backend compilation."""

    spec: KernelSpec
    source: str
    argument_names: tuple[str, ...]
    step_name: str
    step_source: str
    python_function: Callable | None = None
    compiled_function: Callable | None = None

    @property
    def function(self):
        if self.compiled_function is not None:
            return self.compiled_function
        if self.python_function is not None:
            return self.python_function
        raise RuntimeError("Kernel has not been built.")

    @property
    def is_built(self):
        return self.python_function is not None

    @property
    def is_compiled(self):
        return self.compiled_function is not None

    def replace_source(self, source):
        """Replace generated source and invalidate its executable forms."""
        self.source = source
        self.python_function = None
        self.compiled_function = None
