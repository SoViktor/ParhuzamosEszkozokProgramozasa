@echo off

if exist frames (
    del /Q frames\*.ppm
) else (
    mkdir frames
)

gcc ^
-DRUN_CPU=0 ^
-DCL_TARGET_OPENCL_VERSION=120 ^
FluidSim/main.c ^
FluidSim/fluid_sim.c ^
FluidSim/fluid_cpu.c ^
FluidSim/fluid_opencl.c ^
FluidSim/fluid_visualizer.c ^
KernelLoader/KernelLoader.c ^
-o fluid_opencl.exe ^
-IFluidSim ^
-IKernelLoader ^
-I"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" ^
-L"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib" ^
-lOpenCL

fluid_opencl.exe