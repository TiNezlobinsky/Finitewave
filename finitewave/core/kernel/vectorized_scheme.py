import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme, remove_decorators


class VectorizedKernelScheme(KernelScheme):
    """Generate a kernel that does not use explicit loops, but vectorized array operations instead."""

    def generate(self, spec):
        argument_names = self._collect_kernel_arguments(spec)
        signature = ", ".join(argument_names)

        input_setup = self._generate_input_setup(spec)
        observers = self._generate_observers(spec)
        output = self._generate_output(spec)

        source = f"""
        def {spec.name}({signature}):
            {input_setup}

            {spec.step_body}

            {observers}

            {output}
        """

        source = textwrap.dedent(source).strip()

        return KernelArtifact(
            spec=spec,
            source=source,
            argument_names=tuple(argument_names),
        )
