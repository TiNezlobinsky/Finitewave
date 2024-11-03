
"""
finitewave
==========

A Python package for simulating cardiac electrophysiology in 2D and 3D using
the finite difference method.

This package provides a set of tools for simulating cardiac electrophysiology
in 2D and 3D using the finite difference method. The package includes classes
for creating cardiac tissue models, tracking electrical activity, and
visualizing simulation results. The package is designed to be flexible and
extensible, allowing users to create custom models and trackers for their
specific research needs.

"""

from finitewave.core import (
    Command,
    CommandSequence,
    FibrosisPattern,
    CardiacModel,
    StateKeeper,
    Stencil,
    StimCurrent,
    StimSequence,
    StimVoltage,
    Stim,
    CardiacTissue,
    Tracker,
    TrackerSequence
)

from finitewave.cpuwave2D import (
    IncorrectWeightsModeError2D,
    Diffuse2DPattern,
    ScarGauss2DPattern,
    ScarRect2DPattern,
    Structural2DPattern,
    diffuse_kernel_2d_iso,
    diffuse_kernel_2d_aniso,
    _parallel,
    AlievPanfilov2D,
    AlievPanfilovKernels2D,
    LuoRudy912D,
    LuoRudy91Kernels2D,
    TP062D,
    TP06Kernels2D,
    LuoRudy912D,
    LuoRudy91Kernels2D,
    TP062D,
    TP06Kernels2D,
    AsymmetricStencil2D,
    IsotropicStencil2D,
    StimCurrentCoord2D,
    StimVoltageCoord2D,
    StimCurrentMatrix2D,
    StimVoltageMatrix2D,
    CardiacTissue2D,
    ActionPotential2DTracker,
    ActivationTime2DTracker,
    Animation2DTracker,
    ECG2DTracker,
    LocalActivationTime2DTracker,
    MultiVariable2DTracker,
    Period2DTracker,
    PeriodAnimation2DTracker,
    SpiralWaveCore2DTracker,
    Variable2DTracker,
)
from finitewave.cpuwave3D import (
    Diffuse3DPattern,
    Structural3DPattern,
    diffuse_kernel_3d_iso,
    diffuse_kernel_3d_aniso,
    _parallel,
    AlievPanfilov3D,
    AlievPanfilovKernels3D,
    LuoRudy913D,
    LuoRudy91Kernels3D,
    TP063D,
    TP06Kernels3D,
    LuoRudy913D,
    LuoRudy91Kernels3D,
    TP063D,
    TP06Kernels3D,
    AsymmetricStencil3D,
    IsotropicStencil3D,
    StimCurrentCoord3D,
    StimVoltageCoord3D,
    StimCurrentMatrix3D,
    StimVoltageMatrix3D,
    CardiacTissue3D,
    ActionPotential3DTracker,
    ActivationTime3DTracker,
    AnimationSlice3DTracker,
    ECG3DTracker,
    Period3DTracker,
    SpiralWaveCore3DTracker,
    Variable3DTracker,
    MultiVariable3DTracker,
    VTKFrame3DTracker,
    Animation3DTracker,
    PeriodAnimation3DTracker
)

from finitewave.tools import (
    Animation2DBuilder,
    Animation3DBuilder,
    DriftVelocityCalculation,
    PlanarWaveVelocity2DCalculation,
    VisMeshBuilder3D,
)
