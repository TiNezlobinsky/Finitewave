# API Reference

## core

### command

- [Command](./api/finitewave/core/command/command.md)
- [CommandSequence](./api/finitewave/core/command/command_sequence.md)

### exception

- [IncorrectWeightsShapeError](./api/finitewave/core/exception/exceptions.md)

### fibrosis

- [FibrosisPattern](./api/finitewave/core/fibrosis/fibrosis_pattern.md)

### model

- [CardiacModel](./api/finitewave/core/model/cardiac_model.md)

### state

- [StateKeeper](./api/finitewave/core/state/state_keeper.md)

### stencil

- [Stencil](./api/finitewave/core/stencil/stencil.md)

### stimulation

- [Stim](./api/finitewave/core/stimulation/stim.md)
- [StimCurrent](./api/finitewave/core/stimulation/stim_current.md)
- [StimSequence](./api/finitewave/core/stimulation/stim_sequence.md)
- [StimVoltage](./api/finitewave/core/stimulation/stim_voltage.md)

### tissue

- [CardiacTissue](./api/finitewave/core/tissue/cardiac_tissue.md)

### tracker

- [Tracker](./api/finitewave/core/tracker/tracker.md)
- [TrackerSequence](./api/finitewave/core/tracker/tracker_sequence.md)

## cpuwave2D

### exception

- [IncorrectWeightsModeError2D](./api/finitewave/cpuwave2D/exception/exceptions_2d.md)

### fibrosis

- [Diffuse2DPattern](./api/finitewave/cpuwave2D/fibrosis/diffuse_2d_pattern.md)
- [ScarGauss2DPattern](./api/finitewave/cpuwave2D/fibrosis/scar_gauss_2d_pattern.md)
- [ScarRect2DPattern](./api/finitewave/cpuwave2D/fibrosis/scar_rect_2d_pattern.md)
- [Structural2DPattern](./api/finitewave/cpuwave2D/fibrosis/structural_2d_pattern.md)

### model

- [AlievPanfilov2D](./api/finitewave/cpuwave2D/model/aliev_panfilov_2d.md)
- [AlievPanfilovKernels2D](./api/finitewave/cpuwave2D/model/aliev_panfilov_kernels_2d.md)
- [LuoRudy912D](./api/finitewave/cpuwave2D/model/luo_rudy91_2d.md)
- [LuoRudy91Kernels2D](./api/finitewave/cpuwave2D/model/luo_rudy91_kernels_2d.md)
- [TP062D](./api/finitewave/cpuwave2D/model/tp06_2d.md)
- [TP06Kernels2D](./api/finitewave/cpuwave2D/model/tp06_kernels_2d.md)

### stencil

- [AsymmetricStencil2D](./api/finitewave/cpuwave2D/stencil/asymmetric_stencil_2d.md)
- [IsotropicStencil2D](./api/finitewave/cpuwave2D/stencil/isotropic_stencil_2d.md)

### stimulation

- [StimCurrentCoord2D](./api/finitewave/cpuwave2D/stimulation/stim_current_coord_2d.md)
- [StimCurrentMatrix2D](./api/finitewave/cpuwave2D/stimulation/stim_current_matrix_2d.md)
- [StimVoltageCoord2D](./api/finitewave/cpuwave2D/stimulation/stim_voltage_coord_2d.md)
- [StimVoltageMatrix2D](./api/finitewave/cpuwave2D/stimulation/stim_voltage_matrix_2d.md)

### tissue

- [CardiacTissue2D](./api/finitewave/cpuwave2D/tissue/cardiac_tissue_2d.md)

### tracker

- [ActionPotential2DTracker](./api/finitewave/cpuwave2D/tracker/action_potential_2d_tracker.md)
- [ActivationTime2DTracker](./api/finitewave/cpuwave2D/tracker/activation_time_2d_tracker.md)
- [Animation2DTracker](./api/finitewave/cpuwave2D/tracker/animation_2d_tracker.md)
- [ECG2DTracker](./api/finitewave/cpuwave2D/tracker/ecg_2d_tracker.md)
- [MultiActivationTime2DTracker](./api/finitewave/cpuwave2D/tracker/multi_activation_time_2d_tracker.md)
- [MultiVariable2DTracker](./api/finitewave/cpuwave2D/tracker/multi_variable_2d_tracker.md)
- [Period2DTracker](./api/finitewave/cpuwave2D/tracker/period_2d_tracker.md)
- [PeriodMap2DTracker](./api/finitewave/cpuwave2D/tracker/period_map_2d_tracker.md)
- [Spiral2DTracker](./api/finitewave/cpuwave2D/tracker/spiral_2d_tracker.md)
- [Variable2DTracker](./api/finitewave/cpuwave2D/tracker/variable_2d_tracker.md)
- [Velocity2DTracker](./api/finitewave/cpuwave2D/tracker/velocity_2d_tracker.md)

## cpuwave3D

### fibers

- [RotationalAnisotropy](./api/finitewave/cpuwave3D/fibers/rotational_anisotropy.md)

### fibrosis

- [Diffuse3DPattern](./api/finitewave/cpuwave3D/fibrosis/diffuse_3d_pattern.md)
- [Structural3DPattern](./api/finitewave/cpuwave3D/fibrosis/structural_3d_pattern.md)

### model

- [AlievPanfilov3D](./api/finitewave/cpuwave3D/model/aliev_panfilov_3d.md)
- [AlievPanfilovKernels3D](./api/finitewave/cpuwave3D/model/aliev_panfilov_kernels_3d.md)
- [LuoRudy913D](./api/finitewave/cpuwave3D/model/luo_rudy91_3d.md)
- [LuoRudy91Kernels3D](./api/finitewave/cpuwave3D/model/luo_rudy91_kernels_3d.md)
- [TP063D](./api/finitewave/cpuwave3D/model/tp06_3d.md)
- [TP06Kernels3D](./api/finitewave/cpuwave3D/model/tp06_kernels_3d.md)

### stencil

- [AsymmetricStencil3D](./api/finitewave/cpuwave3D/stencil/asymmetric_stencil_3d.md)
- [IsotropicStencil3D](./api/finitewave/cpuwave3D/stencil/isotropic_stencil_3d.md)

### stimulation

- [StimCurrentCoord3D](./api/finitewave/cpuwave3D/stimulation/stim_current_coord_3d.md)
- [StimCurrentMatrix3D](./api/finitewave/cpuwave3D/stimulation/stim_current_matrix_3d.md)
- [StimVoltageCoord3D](./api/finitewave/cpuwave3D/stimulation/stim_voltage_coord_3d.md)
- [StimVoltageMatrix3D](./api/finitewave/cpuwave3D/stimulation/stim_voltage_matrix_3d.md)

### tissue

- [CardiacTissue3D](./api/finitewave/cpuwave3D/tissue/cardiac_tissue_3d.md)

### tracker

- [ActionPotential3DTracker](./api/finitewave/cpuwave3D/tracker/action_potential_3d_tracker.md)
- [ActivationTime3DTracker](./api/finitewave/cpuwave3D/tracker/activation_time_3d_tracker.md)
- [Animation3DTracker](./api/finitewave/cpuwave3D/tracker/animation_3d_tracker.md)
- [AnimationSlice3DTracker](./api/finitewave/cpuwave3D/tracker/animation_slice_3d_tracker.md)
- [ECG3DTracker](./api/finitewave/cpuwave3D/tracker/ecg_3d_tracker.md)
- [Period3DTracker](./api/finitewave/cpuwave3D/tracker/period_3d_tracker.md)
- [PeriodMap3DTracker](./api/finitewave/cpuwave3D/tracker/period_map_3d_tracker.md)
- [Spiral3DTracker](./api/finitewave/cpuwave3D/tracker/spiral_3d_tracker.md)
- [VTKFrame3DTracker](./api/finitewave/cpuwave3D/tracker/vtk_frame_3d_tracker.md)
- [Variable3DTracker](./api/finitewave/cpuwave3D/tracker/variable_3d_tracker.md)
- [Velocity3DTracker](./api/finitewave/cpuwave3D/tracker/velocity_3d_tracker.md)

## tools

- [Animation3DBuilder](./api/finitewave/tools/animation_3d_builder/animation_3d_builder.md)

- [AnimationBuilder](./api/finitewave/tools/animation_builder/animation_builder.md)

- [DriftVelocityCalculation](./api/finitewave/tools/drift_velocity_calculation/drift_velocity_calculation.md)

- [PotentialPeriodAnimationBuilder](./api/finitewave/tools/potential_period_animation_builder/potential_period_animation_builder.md)

- [StimSequence](./api/finitewave/tools/stim_sequence/stim_sequence.md)

- [VisMeshBuilder3D](./api/finitewave/tools/vis_mesh_builder_3d/vis_mesh_builder_3d.md)

- [VTKMeshBuilder](./api/finitewave/tools/vtk_mesh_builder/vtk_mesh_builder.md)

