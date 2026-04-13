__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* result,
                         const int n)
{
    int id = get_global_id(0);

    if (id < n) {
        result[id] = a[id] + b[id];
    }
}