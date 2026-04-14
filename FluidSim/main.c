#include <stdio.h>
#include <stdlib.h>

#include "fluid_sim.h"
#include "fluid_opencl.h"

#define KERNEL_FILE_PATH "FluidSim/Kernels/Fluid_step.cl"
#define KERNEL_NAME "fluid_step"

#define SIMULATION_STEPS 200
#define PRINT_INTERVAL 10

static void build_test_scene(FluidSim* sim)
{
    if (sim == NULL) {
        return;
    }

    fluid_sim_clear(sim);

    /* Kulso falak */
    fluid_sim_add_border_walls(sim);

    /* Also platform */
    fluid_sim_add_rect_solid(sim, 18, 45, 28, 2);

    /* Kozepso akadaly */
    fluid_sim_add_rect_solid(sim, 30, 30, 4, 8);

    /* Bal oldali kisebb platform */
    fluid_sim_add_rect_solid(sim, 10, 38, 10, 2);

    /* Jobb oldali kisebb platform */
    fluid_sim_add_rect_solid(sim, 44, 38, 10, 2);

    /* Kezdeti folyadektomb felul */
    fluid_sim_add_rect_mass(sim, 24, 4, 16, 10, 1.0f);
}

int main(void)
{
    FluidSim sim;
    FluidOpenCL gpu;
    int step;

    if (!fluid_sim_init(&sim)) {
        printf("[ERROR] A FluidSim inicializalasa sikertelen.\n");
        return 1;
    }

    build_test_scene(&sim);

    printf("Kezdeti allapot:\n");
    fluid_sim_print_combined(&sim);
    printf("\n");

    if (!fluid_opencl_init(&gpu, KERNEL_FILE_PATH, KERNEL_NAME)) {
        printf("[ERROR] Az OpenCL inicializalasa sikertelen.\n");
        fluid_sim_free(&sim);
        return 1;
    }

    fluid_opencl_set_work_sizes(&gpu, 8, 8);

    if (!fluid_opencl_create_buffers(&gpu, &sim)) {
        printf("[ERROR] Az OpenCL bufferek letrehozasa sikertelen.\n");
        fluid_opencl_free(&gpu);
        fluid_sim_free(&sim);
        return 1;
    }

    if (!fluid_opencl_write_simulation(&gpu, &sim)) {
        printf("[ERROR] A szimulacios adatok feltoltese sikertelen.\n");
        fluid_opencl_free(&gpu);
        fluid_sim_free(&sim);
        return 1;
    }

    for (step = 1; step <= SIMULATION_STEPS; ++step) {
        if (!fluid_opencl_step(&gpu, &sim)) {
            printf("[ERROR] A(z) %d. szimulacios lepes sikertelen.\n", step);
            fluid_opencl_free(&gpu);
            fluid_sim_free(&sim);
            return 1;
        }

        if (step % PRINT_INTERVAL == 0 || step == SIMULATION_STEPS) {
            if (!fluid_opencl_read_mass(&gpu, &sim)) {
                printf("[ERROR] A(z) %d. lepes utani visszaolvasas sikertelen.\n", step);
                fluid_opencl_free(&gpu);
                fluid_sim_free(&sim);
                return 1;
            }

            printf("Allapot %d lepes utan:\n", step);
            fluid_sim_print_combined(&sim);
            printf("\n");
        }
    }

    fluid_opencl_free(&gpu);
    fluid_sim_free(&sim);

    return 0;
}