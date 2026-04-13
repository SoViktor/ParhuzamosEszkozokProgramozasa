#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vector.h"

#define N 1000
#define EPSILON 0.0001f

static void vector_add_sequential(const float* a, const float* b, float* result, int n)
{
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

static int verify_results(const float* gpu_result, const float* cpu_result, int n)
{
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > EPSILON) {
            printf("[HIBA] Eltérés a(z) %d. indexnél:\n", i);
            printf("  GPU eredmény: %.6f\n", gpu_result[i]);
            printf("  CPU eredmény: %.6f\n", cpu_result[i]);
            printf("  Különbség   : %.6f\n", diff);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    float a[N], b[N], gpu_result[N], cpu_result[N];

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
        gpu_result[i] = 0.0f;
        cpu_result[i] = 0.0f;
    }

    vector_add(a, b, gpu_result, N);

    vector_add_sequential(a, b, cpu_result, N);

    printf("Elso 10 eredmeny:\n");
    for (int i = 0; i < 10; i++) {
        printf("a[%d] = %.2f, b[%d] = %.2f, gpu = %.2f, cpu = %.2f\n",
               i, a[i], i, b[i], gpu_result[i], cpu_result[i]);
    }

    if (verify_results(gpu_result, cpu_result, N)) {
        printf("\n[OK] Az OpenCL eredmeny megegyezik a szekvencialis eredmennyel.\n");
    } else {
        printf("\n[HIBA] Az OpenCL eredmeny NEM egyezik meg a szekvencialis eredmennyel.\n");
    }

    return 0;
}