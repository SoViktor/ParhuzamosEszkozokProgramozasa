#ifndef FLUID_OPENCL_H
#define FLUID_OPENCL_H

#include <CL/cl.h>

#include "fluid_sim.h"

typedef struct FluidOpenCL {
    cl_platform_id platform_id;
    cl_device_id device_id;

    cl_context context;
    cl_command_queue command_queue;

    cl_program program;
    cl_kernel kernel;

    cl_mem mass_buffer;
    cl_mem next_mass_buffer;
    cl_mem solid_buffer;

    size_t global_work_size[2];
    size_t local_work_size[2];
} FluidOpenCL;

int fluid_opencl_init(FluidOpenCL* gpu, const char* kernel_file_path, const char* kernel_name);
void fluid_opencl_free(FluidOpenCL* gpu);

int fluid_opencl_create_buffers(FluidOpenCL* gpu, const FluidSim* sim);
void fluid_opencl_release_buffers(FluidOpenCL* gpu);

int fluid_opencl_write_simulation(FluidOpenCL* gpu, const FluidSim* sim);
int fluid_opencl_read_mass(FluidOpenCL* gpu, FluidSim* sim);
int fluid_opencl_read_next_mass(FluidOpenCL* gpu, FluidSim* sim);

int fluid_opencl_set_kernel_args(FluidOpenCL* gpu, const FluidSim* sim);
int fluid_opencl_step(FluidOpenCL* gpu, FluidSim* sim);

void fluid_opencl_swap_buffers(FluidOpenCL* gpu);
void fluid_opencl_set_work_sizes(FluidOpenCL* gpu, size_t local_x, size_t local_y);

#endif