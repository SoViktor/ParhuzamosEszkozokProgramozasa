#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef RUN_CPU
#define RUN_CPU 1
#endif

#ifndef RUN_COMPARE
#define RUN_COMPARE 0
#endif

#ifndef SAVE_FRAMES
#define SAVE_FRAMES 1
#endif

#include "fluid_sim.h"
#include "fluid_visualizer.h"
#include "fluid_cpu.h"
#if RUN_COMPARE || !RUN_CPU
#include "fluid_opencl.h"
#endif

#define KERNEL_FILE_PATH "FluidSim/Kernels/Fluid_step.cl"
#define KERNEL_NAME "fluid_step"

#define SIMULATION_STEPS 200
#define PRINT_INTERVAL 1
#define FRAME_SCALE 8
#define EPSILON 0.00001f

static void build_test_scene(FluidSim* sim)
{
    if (sim == NULL) {
        return;
    }

    fluid_sim_clear(sim);
    fluid_sim_add_border_walls(sim);
    fluid_sim_add_rect_solid(sim, 18, 45, 28, 2);
    fluid_sim_add_rect_solid(sim, 30, 30, 4, 8);
    fluid_sim_add_rect_solid(sim, 10, 38, 10, 2);
    fluid_sim_add_rect_solid(sim, 44, 38, 10, 2);
    fluid_sim_add_rect_mass(sim, 24, 4, 16, 10, 1.0f);
}

static double seconds_since(clock_t start, clock_t end)
{
    return (double)(end - start) / (double)CLOCKS_PER_SEC;
}

static float simulation_checksum(const FluidSim* sim)
{
    int i;
    float sum = 0.0f;

    if (sim == NULL) {
        return 0.0f;
    }

    for (i = 0; i < SIM_CELL_COUNT; ++i) {
        sum += sim->mass[i];
    }

    return sum;
}

static float max_abs_difference(const FluidSim* a, const FluidSim* b)
{
    int i;
    float diff;
    float max_diff = 0.0f;

    if (a == NULL || b == NULL) {
        return -1.0f;
    }

    for (i = 0; i < SIM_CELL_COUNT; ++i) {
        diff = fabsf(a->mass[i] - b->mass[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

#if RUN_COMPARE || !RUN_CPU
static int init_opencl_runner(FluidOpenCL* gpu, FluidSim* sim)
{
    if (!fluid_opencl_init(gpu, KERNEL_FILE_PATH, KERNEL_NAME)) {
        printf("[ERROR] Az OpenCL inicializalasa sikertelen.\n");
        return 0;
    }

    fluid_opencl_set_work_sizes(gpu, 8, 8);

    if (!fluid_opencl_create_buffers(gpu, sim)) {
        printf("[ERROR] Az OpenCL bufferek letrehozasa sikertelen.\n");
        fluid_opencl_free(gpu);
        return 0;
    }

    if (!fluid_opencl_write_simulation(gpu, sim)) {
        printf("[ERROR] A szimulacios adatok feltoltese sikertelen.\n");
        fluid_opencl_free(gpu);
        return 0;
    }

    return 1;
}
#endif

#if RUN_COMPARE
static int run_compare_mode(void)
{
    FluidSim cpu_sim;
    FluidSim gpu_sim;
    FluidOpenCL gpu;
    int step;
    clock_t start_time;
    clock_t end_time;
    double cpu_seconds;
    double opencl_seconds;
    float max_diff;

    if (!fluid_sim_init(&cpu_sim) || !fluid_sim_init(&gpu_sim)) {
        printf("[ERROR] A FluidSim inicializalasa sikertelen.\n");
        fluid_sim_free(&cpu_sim);
        fluid_sim_free(&gpu_sim);
        return 1;
    }

    build_test_scene(&cpu_sim);
    build_test_scene(&gpu_sim);

    if (!init_opencl_runner(&gpu, &gpu_sim)) {
        fluid_sim_free(&cpu_sim);
        fluid_sim_free(&gpu_sim);
        return 1;
    }

    start_time = clock();
    for (step = 1; step <= SIMULATION_STEPS; ++step) {
        if (!fluid_cpu_step(&cpu_sim)) {
            printf("[ERROR] CPU szimulacios lepes sikertelen.\n");
            fluid_opencl_free(&gpu);
            fluid_sim_free(&cpu_sim);
            fluid_sim_free(&gpu_sim);
            return 1;
        }
    }
    end_time = clock();
    cpu_seconds = seconds_since(start_time, end_time);

    start_time = clock();
    for (step = 1; step <= SIMULATION_STEPS; ++step) {
        if (!fluid_opencl_step(&gpu, &gpu_sim)) {
            printf("[ERROR] OpenCL szimulacios lepes sikertelen.\n");
            fluid_opencl_free(&gpu);
            fluid_sim_free(&cpu_sim);
            fluid_sim_free(&gpu_sim);
            return 1;
        }
    }
    end_time = clock();
    opencl_seconds = seconds_since(start_time, end_time);

    if (!fluid_opencl_read_mass(&gpu, &gpu_sim)) {
        printf("[ERROR] OpenCL vegeredmeny visszaolvasasa sikertelen.\n");
        fluid_opencl_free(&gpu);
        fluid_sim_free(&cpu_sim);
        fluid_sim_free(&gpu_sim);
        return 1;
    }

    max_diff = max_abs_difference(&cpu_sim, &gpu_sim);

    printf("\n[COMPARE] Lepesszam: %d\n", SIMULATION_STEPS);
    printf("[COMPARE] CPU ido:    %.6f s\n", cpu_seconds);
    printf("[COMPARE] OpenCL ido: %.6f s\n", opencl_seconds);
    if (opencl_seconds > 0.0) {
        printf("[COMPARE] Gyorsitas:  %.3fx\n", cpu_seconds / opencl_seconds);
    }
    printf("[COMPARE] CPU checksum:    %.9f\n", simulation_checksum(&cpu_sim));
    printf("[COMPARE] OpenCL checksum: %.9f\n", simulation_checksum(&gpu_sim));
    printf("[COMPARE] Max elteres: %.9f\n", max_diff);

    if (max_diff <= EPSILON) {
        printf("[COMPARE] Eredmeny: OK, a ket verzio azonos tolerancian belul.\n");
    } else {
        printf("[COMPARE] Eredmeny: ELTERES, tovabbi vizsgalat kell.\n");
    }

    fluid_save_ppm(&cpu_sim, "frames/final_cpu.ppm", FRAME_SCALE);
    fluid_save_ppm(&gpu_sim, "frames/final_opencl.ppm", FRAME_SCALE);

    fluid_opencl_free(&gpu);
    fluid_sim_free(&cpu_sim);
    fluid_sim_free(&gpu_sim);

    return 0;
}
#else
static int run_single_mode(void)
{
    FluidSim sim;
#if !RUN_CPU
    FluidOpenCL gpu;
#endif
    int step;
    char filename[128];
    clock_t start_time;
    clock_t end_time;
    double seconds;

    if (!fluid_sim_init(&sim)) {
        printf("[ERROR] A FluidSim inicializalasa sikertelen.\n");
        return 1;
    }

    build_test_scene(&sim);

#if !RUN_CPU
    if (!init_opencl_runner(&gpu, &sim)) {
        fluid_sim_free(&sim);
        return 1;
    }
#endif

    start_time = clock();

    for (step = 1; step <= SIMULATION_STEPS; ++step) {
#if RUN_CPU
        if (!fluid_cpu_step(&sim)) {
            printf("[ERROR] A(z) %d. CPU szimulacios lepes sikertelen.\n", step);
            fluid_sim_free(&sim);
            return 1;
        }
#else
        if (!fluid_opencl_step(&gpu, &sim)) {
            printf("[ERROR] A(z) %d. OpenCL szimulacios lepes sikertelen.\n", step);
            fluid_opencl_free(&gpu);
            fluid_sim_free(&sim);
            return 1;
        }
#endif

#if SAVE_FRAMES
        if (step % PRINT_INTERVAL == 0 || step == SIMULATION_STEPS) {
#if !RUN_CPU
            if (!fluid_opencl_read_mass(&gpu, &sim)) {
                printf("[ERROR] A(z) %d. lepes utani visszaolvasas sikertelen.\n", step);
                fluid_opencl_free(&gpu);
                fluid_sim_free(&sim);
                return 1;
            }
#endif
            sprintf(filename, "frames/frame_%05d.ppm", step);
            if (!fluid_save_ppm(&sim, filename, FRAME_SCALE)) {
                printf("[ERROR] Frame mentes sikertelen: %s\n", filename);
            }
        }
#endif
    }

#if !RUN_CPU
#if !SAVE_FRAMES
    if (!fluid_opencl_read_mass(&gpu, &sim)) {
        printf("[ERROR] OpenCL vegeredmeny visszaolvasasa sikertelen.\n");
        fluid_opencl_free(&gpu);
        fluid_sim_free(&sim);
        return 1;
    }
#endif
#endif

    end_time = clock();
    seconds = seconds_since(start_time, end_time);

#if RUN_CPU
    printf("[INFO] CPU futasi ido: %.6f s\n", seconds);
#else
    printf("[INFO] OpenCL futasi ido: %.6f s\n", seconds);
    fluid_opencl_free(&gpu);
#endif

    printf("[INFO] Checksum: %.9f\n", simulation_checksum(&sim));
    fluid_sim_free(&sim);

    return 0;
}
#endif

int main(void)
{
#if RUN_COMPARE
    return run_compare_mode();
#else
    return run_single_mode();
#endif
}
