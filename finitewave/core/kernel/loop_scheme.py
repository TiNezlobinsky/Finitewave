import textwrap

from .artifact import KernelArtifact
from .scheme import KernelScheme, remove_decorators


class LoopKernelScheme(KernelScheme):
    """Generate a scalar kernel with an explicit loop over tissue indexes."""

    def __init__(self, loop_function: str = "kernel_range"):
        self.loop_function = loop_function

    def generate(self, spec):
        argument_names = self._collect_kernel_arguments(spec)
        signature = ", ".join(argument_names)

        input_setup = self._generate_input_setup(spec)
        local_vars = self._generate_local_variables(spec)
        updates = self._generate_state_updates(spec)
        observers = self._generate_observers(spec)
        output = self._generate_output(spec)

        source = f"""
        def {spec.name}({signature}):
            {input_setup}

            for index in {self.loop_function}(len(indexes)):
                {local_vars}

                {spec.step_body}

                {updates}
                {observers}

            {output}
        """

        source = textwrap.dedent(source).strip()

        return KernelArtifact(
            spec=spec,
            source=source,
            argument_names=tuple(argument_names),
        )

    def _generate_local_variables(self, spec):
        lines = []

        for name in spec.arrays:
            if name == "rhs":
                continue

            lines.append(f"{name} = {name}_[index]")

        return self._indent(lines)

    def _generate_state_updates(self, spec):
        lines = []

        for name in spec.state_vars:
            if name == "u":
                continue

            lines.append(f"{name}_[index] = {name}_new")

        lines.append("rhs_[index] = rhs")

        return self._indent(lines)
