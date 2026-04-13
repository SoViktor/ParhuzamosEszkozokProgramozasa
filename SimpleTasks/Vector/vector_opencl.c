#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#include "KernelLoader.h"
#include "vector.h"

void vector_add(float* a, float* b, float* result, int n)
{
    cl_int err;

    char* kernelCode = loadKernelSource("SimpleTasks/Vector/Kernels/VectorKernel.cl");
    if (kernelCode == NULL) return;

    cl_platform_id platform_id;
    clGetPlatformIDs(1, &platform_id, NULL);

    cl_device_id device_id;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, NULL, NULL);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

    cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, NULL);
    cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, NULL);
    cl_mem device_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, NULL);

    clEnqueueWriteBuffer(queue, device_a, CL_FALSE, 0, n * sizeof(float), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, device_b, CL_FALSE, 0, n * sizeof(float), b, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_result);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t local = 256;
    size_t global = ((n + local - 1) / local) * local;

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, device_result, CL_TRUE, 0, n * sizeof(float), result, 0, NULL, NULL);

    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_result);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(kernelCode);
}