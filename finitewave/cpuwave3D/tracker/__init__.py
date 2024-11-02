"""
## 3D Tracker

This module contains classes for tracking the evolution of the wavefront in 3D.
Most of the classes in this module are similar to the ones in the 2D tracker
module. More information can be found in the documentation for the 2D tracker
module.

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

1. `ActionPotential3DTracker` - Tracks the action potential in 3D.
2. `ActivationTime3DTracker` - Tracks the activation time in 3D.
3. `AnimationSlice3DTracker` - Tracks the animation slice in 3D.
4. `ECG3DTracker` - Tracks the ECG in 3D.
5. `Period3DTracker` - Tracks the period in 3D.
6. `PeriodAnimation3DTracker` - Tracks the period animation in 3D.
7. `SpiralWaveCore3DTracker` - Tracks the spiral wave core in 3D.
8. `Variable3DTracker` - Tracks a variable in 3D.
9. `MultiVariable3DTracker` - Tracks multiple variables in 3D.
10. `VTKFrame3DTracker` - Tracks the VTK frame in 3D.
11. `Animation3DTracker` - Tracks the animation in 3D.
"""

from .action_potential_3d_tracker import ActionPotential3DTracker
from .activation_time_3d_tracker import ActivationTime3DTracker
from .animation_slice_3d_tracker import AnimationSlice3DTracker
from .ecg_3d_tracker import ECG3DTracker
from .period_3d_tracker import Period3DTracker
from .period_animation_3d_tracker import PeriodAnimation3DTracker
from .spiral_wave_core_3d_tracker import SpiralWaveCore3DTracker
from .variable_3d_tracker import Variable3DTracker
from .multi_variable_3d_tracker import MultiVariable3DTracker
from .vtk_frame_3d_tracker import VTKFrame3DTracker
from .animation_3d_tracker import Animation3DTracker

__all__ = [
    "ActionPotential3DTracker",
    "ActivationTime3DTracker",
    "AnimationSlice3DTracker",
    "ECG3DTracker",
    "Period3DTracker",
    "PeriodAnimation3DTracker",
    "SpiralWaveCore3DTracker",
    "Variable3DTracker",
    "MultiVariable3DTracker",
    "VTKFrame3DTracker",
    "Animation3DTracker",
]
