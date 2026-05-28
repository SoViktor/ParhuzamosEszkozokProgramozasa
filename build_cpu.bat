@echo off

if exist frames (
    del /Q frames\*.ppm
) else (
    mkdir frames
)

gcc ^
-DRUN_CPU=1 ^
-DSAVE_FRAMES=1 ^
FluidSim/main.c ^
FluidSim/fluid_sim.c ^
FluidSim/fluid_cpu.c ^
FluidSim/fluid_visualizer.c ^
-o fluid_cpu.exe ^
-IFluidSim

fluid_cpu.exe
