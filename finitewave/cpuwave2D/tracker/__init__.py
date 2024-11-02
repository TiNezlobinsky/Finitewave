"""
## 2D Tracker

This module contains classes for tracking the evolution of the wavefront in 2D.

Each tracker class has basic attributes such as `start_time`, `end_time`,
`step`, `path`, and `file_name`. Note that the `start_time` and `end_time`
is given in time units, and the `step` is the number of time steps between
recordings.

The tracker classes can be grouped into the following categories:

1. Full field trackers: These trackers track the entire field and output the
    results in a single array.
2. Point trackers: These trackers track the evolution of a specific point(s)
    in the field.
3. Animation trackers: These trackers track the evolution of the field over
    time and save the results as frames for creating animations.

### List of classes in this module:

1. `ActionPotential2DTracker` - Tracks the action potential in 2D.
2. `ActivationTime2DTracker` - Tracks the activation time in 2D.
3. `Animation2DTracker` - Tracks the animation in 2D.
4. `ECG2DTracker` - Tracks the ECG in 2D.
5. `LocalActivationTime2DTracker` - Tracks the local activation time in 2D.
6. `MultiVariable2DTracker` - Tracks multiple variables in 2D.
7. `Period2DTracker` - Tracks the period in 2D.
8. `PeriodAnimation2DTracker` - Tracks the period animation in 2D.
9. `SpiralWaveCore2DTracker` - Tracks the spiral wave core in 2D.
10. `Variable2DTracker` - Tracks a variable in 2D.
"""

from .action_potential_2d_tracker import ActionPotential2DTracker
from .activation_time_2d_tracker import ActivationTime2DTracker
from .animation_2d_tracker import Animation2DTracker
from .ecg_2d_tracker import ECG2DTracker
from .local_activation_time_2d_tracker import LocalActivationTime2DTracker
from .multi_variable_2d_tracker import MultiVariable2DTracker
from .period_2d_tracker import Period2DTracker
from .period_animation_2d_tracker import PeriodAnimation2DTracker
from .spiral_wave_core_2d_tracker import SpiralWaveCore2DTracker
from .variable_2d_tracker import Variable2DTracker

__all__ = [
    "ActionPotential2DTracker",
    "ActivationTime2DTracker",
    "Animation2DTracker",
    "ECG2DTracker",
    "LocalActivationTime2DTracker",
    "MultiVariable2DTracker",
    "Period2DTracker",
    "PeriodAnimation2DTracker",
    "SpiralWaveCore2DTracker",
    "Variable2DTracker",
]
