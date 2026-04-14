#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "fluid_opencl.h"
#include "KernelLoader.h"

static void fluid_opencl_reset(FluidOpenCL* gpu)
{
    if (gpu == NULL) {
        return;
    }

    gpu->platform_id = NULL;
    gpu->device_id = NULL;

    gpu->context = NULL;
    gpu->command_queue = NULL;

    gpu->program = NULL;
    gpu->kernel = NULL;

    gpu->mass_buffer = NULL;
    gpu->next_mass_buffer = NULL;
    gpu->solid_buffer = NULL;

    gpu->global_work_size[0] = SIM_WIDTH;
    gpu->global_work_size[1] = SIM_HEIGHT;

    gpu->local_work_size[0] = 8;
    gpu->local_work_size[1] = 8;
}

void fluid_opencl_set_work_sizes(FluidOpenCL* gpu, size_t local_x, size_t local_y)
{
    if (gpu == NULL) {
        return;
    }

    gpu->global_work_size[0] = SIM_WIDTH;
    gpu->global_work_size[1] = SIM_HEIGHT;

    gpu->local_work_size[0] = local_x;
    gpu->local_work_size[1] = local_y;
}

int fluid_opencl_init(FluidOpenCL* gpu, const char* kernel_file_path, const char* kernel_name)
{
    cl_int err;
    cl_uint n_platforms;
    cl_uint n_devices;
    char* kernel_code;
    const char* source;

    if (gpu == NULL || kernel_file_path == NULL || kernel_name == NULL) {
        return 0;
    }

    fluid_opencl_reset(gpu);

    kernel_code = loadKernelSource(kernel_file_path);
    if (kernel_code == NULL) {
        printf("[ERROR] Nem sikerult betolteni a kernel fajlt: %s\n", kernel_file_path);
        return 0;
    }

    err = clGetPlatformIDs(1, &gpu->platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetPlatformIDs sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        return 0;
    }

    err = clGetDeviceIDs(
        gpu->platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &gpu->device_id,
        &n_devices
    );

    if (err != CL_SUCCESS) {
        printf("[ERROR] clGetDeviceIDs sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        return 0;
    }

    gpu->context = clCreateContext(NULL, 1, &gpu->device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS || gpu->context == NULL) {
        printf("[ERROR] clCreateContext sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        return 0;
    }

    gpu->command_queue = clCreateCommandQueue(gpu->context, gpu->device_id, 0, &err);
    if (err != CL_SUCCESS || gpu->command_queue == NULL) {
        printf("[ERROR] clCreateCommandQueue sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        fluid_opencl_free(gpu);
        return 0;
    }

    source = kernel_code;
    gpu->program = clCreateProgramWithSource(gpu->context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS || gpu->program == NULL) {
        printf("[ERROR] clCreateProgramWithSource sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        fluid_opencl_free(gpu);
        return 0;
    }

    err = clBuildProgram(gpu->program, 1, &gpu->device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        char* build_log = NULL;

        printf("[ERROR] clBuildProgram sikertelen. Hibakod: %d\n", err);

        clGetProgramBuildInfo(
            gpu->program,
            gpu->device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &log_size
        );

        if (log_size > 0) {
            build_log = (char*)malloc(log_size + 1);
            if (build_log != NULL) {
                clGetProgramBuildInfo(
                    gpu->program,
                    gpu->device_id,
                    CL_PROGRAM_BUILD_LOG,
                    log_size,
                    build_log,
                    NULL
                );
                build_log[log_size] = '\0';
                printf("Build log:\n%s\n", build_log);
                free(build_log);
            }
        }

        free(kernel_code);
        fluid_opencl_free(gpu);
        return 0;
    }

    gpu->kernel = clCreateKernel(gpu->program, kernel_name, &err);
    if (err != CL_SUCCESS || gpu->kernel == NULL) {
        printf("[ERROR] clCreateKernel sikertelen. Hibakod: %d\n", err);
        free(kernel_code);
        fluid_opencl_free(gpu);
        return 0;
    }

    free(kernel_code);
    return 1;
}

void fluid_opencl_release_buffers(FluidOpenCL* gpu)
{
    if (gpu == NULL) {
        return;
    }

    if (gpu->mass_buffer != NULL) {
        clReleaseMemObject(gpu->mass_buffer);
        gpu->mass_buffer = NULL;
    }

    if (gpu->next_mass_buffer != NULL) {
        clReleaseMemObject(gpu->next_mass_buffer);
        gpu->next_mass_buffer = NULL;
    }

    if (gpu->solid_buffer != NULL) {
        clReleaseMemObject(gpu->solid_buffer);
        gpu->solid_buffer = NULL;
    }
}

void fluid_opencl_free(FluidOpenCL* gpu)
{
    if (gpu == NULL) {
        return;
    }

    fluid_opencl_release_buffers(gpu);

    if (gpu->kernel != NULL) {
        clReleaseKernel(gpu->kernel);
        gpu->kernel = NULL;
    }

    if (gpu->program != NULL) {
        clReleaseProgram(gpu->program);
        gpu->program = NULL;
    }

    if (gpu->command_queue != NULL) {
        clReleaseCommandQueue(gpu->command_queue);
        gpu->command_queue = NULL;
    }

    if (gpu->context != NULL) {
        clReleaseContext(gpu->context);
        gpu->context = NULL;
    }

    gpu->platform_id = NULL;
    gpu->device_id = NULL;
}

int fluid_opencl_create_buffers(FluidOpenCL* gpu, const FluidSim* sim)
{
    cl_int err;
    size_t mass_size;
    size_t solid_size;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    fluid_opencl_release_buffers(gpu);

    mass_size = (size_t)(sim->width * sim->height) * sizeof(float);
    solid_size = (size_t)(sim->width * sim->height) * sizeof(unsigned char);

    gpu->mass_buffer = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE, mass_size, NULL, &err);
    if (err != CL_SUCCESS || gpu->mass_buffer == NULL) {
        printf("[ERROR] mass_buffer letrehozas sikertelen. Hibakod: %d\n", err);
        fluid_opencl_release_buffers(gpu);
        return 0;
    }

    gpu->next_mass_buffer = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE, mass_size, NULL, &err);
    if (err != CL_SUCCESS || gpu->next_mass_buffer == NULL) {
        printf("[ERROR] next_mass_buffer letrehozas sikertelen. Hibakod: %d\n", err);
        fluid_opencl_release_buffers(gpu);
        return 0;
    }

    gpu->solid_buffer = clCreateBuffer(gpu->context, CL_MEM_READ_ONLY, solid_size, NULL, &err);
    if (err != CL_SUCCESS || gpu->solid_buffer == NULL) {
        printf("[ERROR] solid_buffer letrehozas sikertelen. Hibakod: %d\n", err);
        fluid_opencl_release_buffers(gpu);
        return 0;
    }

    return 1;
}

int fluid_opencl_write_simulation(FluidOpenCL* gpu, const FluidSim* sim)
{
    cl_int err;
    size_t mass_size;
    size_t solid_size;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    mass_size = (size_t)(sim->width * sim->height) * sizeof(float);
    solid_size = (size_t)(sim->width * sim->height) * sizeof(unsigned char);

    err = clEnqueueWriteBuffer(
        gpu->command_queue,
        gpu->mass_buffer,
        CL_TRUE,
        0,
        mass_size,
        sim->mass,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] mass_buffer feltoltese sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    err = clEnqueueWriteBuffer(
        gpu->command_queue,
        gpu->next_mass_buffer,
        CL_TRUE,
        0,
        mass_size,
        sim->next_mass,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] next_mass_buffer feltoltese sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    err = clEnqueueWriteBuffer(
        gpu->command_queue,
        gpu->solid_buffer,
        CL_TRUE,
        0,
        solid_size,
        sim->solid,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] solid_buffer feltoltese sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    return 1;
}

int fluid_opencl_read_mass(FluidOpenCL* gpu, FluidSim* sim)
{
    cl_int err;
    size_t mass_size;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    mass_size = (size_t)(sim->width * sim->height) * sizeof(float);

    err = clEnqueueReadBuffer(
        gpu->command_queue,
        gpu->mass_buffer,
        CL_TRUE,
        0,
        mass_size,
        sim->mass,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] mass_buffer visszaolvasas sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    return 1;
}

int fluid_opencl_read_next_mass(FluidOpenCL* gpu, FluidSim* sim)
{
    cl_int err;
    size_t mass_size;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    mass_size = (size_t)(sim->width * sim->height) * sizeof(float);

    err = clEnqueueReadBuffer(
        gpu->command_queue,
        gpu->next_mass_buffer,
        CL_TRUE,
        0,
        mass_size,
        sim->next_mass,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] next_mass_buffer visszaolvasas sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    return 1;
}

int fluid_opencl_set_kernel_args(FluidOpenCL* gpu, const FluidSim* sim)
{
    cl_int err;
    int width;
    int height;
    float max_mass;
    float min_mass;
    float down_rate;
    float diag_rate;
    float side_rate;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    width = sim->width;
    height = sim->height;
    max_mass = MAX_MASS;
    min_mass = MIN_MASS;
    down_rate = DOWN_RATE;
    diag_rate = DIAG_RATE;
    side_rate = SIDE_RATE;

    err = clSetKernelArg(gpu->kernel, 0, sizeof(cl_mem), (void*)&gpu->mass_buffer);
    err |= clSetKernelArg(gpu->kernel, 1, sizeof(cl_mem), (void*)&gpu->next_mass_buffer);
    err |= clSetKernelArg(gpu->kernel, 2, sizeof(cl_mem), (void*)&gpu->solid_buffer);
    err |= clSetKernelArg(gpu->kernel, 3, sizeof(int), (void*)&width);
    err |= clSetKernelArg(gpu->kernel, 4, sizeof(int), (void*)&height);
    err |= clSetKernelArg(gpu->kernel, 5, sizeof(float), (void*)&max_mass);
    err |= clSetKernelArg(gpu->kernel, 6, sizeof(float), (void*)&min_mass);
    err |= clSetKernelArg(gpu->kernel, 7, sizeof(float), (void*)&down_rate);
    err |= clSetKernelArg(gpu->kernel, 8, sizeof(float), (void*)&diag_rate);
    err |= clSetKernelArg(gpu->kernel, 9, sizeof(float), (void*)&side_rate);

    if (err != CL_SUCCESS) {
        printf("[ERROR] Kernel argumentumok beallitasa sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    return 1;
}

void fluid_opencl_swap_buffers(FluidOpenCL* gpu)
{
    cl_mem temp;

    if (gpu == NULL) {
        return;
    }

    temp = gpu->mass_buffer;
    gpu->mass_buffer = gpu->next_mass_buffer;
    gpu->next_mass_buffer = temp;
}

int fluid_opencl_step(FluidOpenCL* gpu, FluidSim* sim)
{
    cl_int err;

    if (gpu == NULL || sim == NULL) {
        return 0;
    }

    if (!fluid_opencl_set_kernel_args(gpu, sim)) {
        return 0;
    }

    err = clEnqueueNDRangeKernel(
        gpu->command_queue,
        gpu->kernel,
        2,
        NULL,
        gpu->global_work_size,
        gpu->local_work_size,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] Kernel futtatas sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    err = clFinish(gpu->command_queue);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clFinish sikertelen. Hibakod: %d\n", err);
        return 0;
    }

    fluid_opencl_swap_buffers(gpu);
    fluid_sim_swap_buffers(sim);

    return 1;
}