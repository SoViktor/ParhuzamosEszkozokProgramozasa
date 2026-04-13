#include <stdio.h>
#include <stdlib.h>
#include <Cl/cl.h>

#include "KernelLoader.h"

const int SAMPLE_SIZE = 1000;

int main (void)
{
    int i;
    cl_int err;

    char* kernelCode = loadKernelSource("SimpleTasks/Vector/Kernels/VectorKernel.cl");
    if (kernelCode == NULL)
    {
        return 1;
    }

    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        free(kernelCode);
        return 1;
    }

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices
    );
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        free(kernelCode);
        return 1;
    }

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL,NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error creating context. Error code: %d\n", err);
        free(kernelCode);
        return 1;
    }

    const char* source = kernelCode;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Error creating program. Error code: %d\n", err);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("[ERROR] Build error. Code: %d\n", err);

        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* build_log = (char*)malloc(log_size + 1);
        if (build_log != NULL)
        {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
            build_log[log_size] = '\0';
            printf("\n=== Buil log ===\n%s\n", build_log);
            free(build_log);
        }
        
        clReleaseProgram(program);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating kernel. Error code: %d\n", err);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }

    // Host oldali vektorok létrehozása
    float* host_a = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    float* host_b = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    float* host_result = (float*)malloc(SAMPLE_SIZE * sizeof(float));

    if (host_a == NULL || host_b == NULL || host_result == NULL) {
        printf("[ERROR] Host memory allocation failed.\n");
        free(host_a);
        free(host_b);
        free(host_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }

    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_a[i] = (float)i * 1.0f;
        host_b[i] = (float)i * 2.0f;
        host_result[i] = 0.0f;
    }

    cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE * sizeof(float), NULL, &err);
    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, SAMPLE_SIZE * sizeof(float), NULL, &err);
    cl_mem device_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SAMPLE_SIZE * sizeof(float), NULL, &err);

    if (device_a == NULL || device_b == NULL || device_result == NULL) {
        printf("[ERROR] Error creating device buffers.\n");
        free(host_a);
        free(host_b);
        free(host_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_result);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&SAMPLE_SIZE);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error creating command queue. Error code: %d\n", err);
        clReleaseMemObject(device_a);
        clReleaseMemObject(device_b);
        clReleaseMemObject(device_result);
        free(host_a);
        free(host_b);
        free(host_result);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(kernelCode);
        return 1;
    }

    clEnqueueWriteBuffer(
        command_queue,
        device_a,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_a,
        0,
        NULL,
        NULL
    );

    clEnqueueWriteBuffer(
        command_queue,
        device_b,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_b,
        0,
        NULL,
        NULL
    );

    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size - 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    err = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error running kernel. Error code: %d\n", err);
    }

    clEnqueueReadBuffer(
        command_queue,
        device_result,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_result,
        0,
        NULL,
        NULL
    );

    printf("Vector addition results:\n");
    for (i = 0; i < 20; ++i) {
        printf("A[%d] = %.2f, B[%d] = %.2f, Result[%d] = %.2f\n",
               i, host_a[i], i, host_b[i], i, host_result[i]);
    }

    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_result);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(host_a);
    free(host_b);
    free(host_result);
    free(kernelCode);

    return 0;   
}