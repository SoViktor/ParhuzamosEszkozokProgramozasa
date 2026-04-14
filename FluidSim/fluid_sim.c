#include <stdio.h>
#include <stdlib.h>

#include "fluid_sim.h"

int fluid_sim_init(FluidSim* sim)
{
    if (sim == NULL) {
        return 0;
    }

    sim->width = SIM_WIDTH;
    sim->height = SIM_HEIGHT;

    sim->mass = (float*)calloc(SIM_CELL_COUNT, sizeof(float));
    sim->next_mass = (float*)calloc(SIM_CELL_COUNT, sizeof(float));
    sim->solid = (unsigned char*)calloc(SIM_CELL_COUNT, sizeof(unsigned char));

    if (sim->mass == NULL || sim->next_mass == NULL || sim->solid == NULL) {
        fluid_sim_free(sim);
        return 0;
    }

    return 1;
}

void fluid_sim_free(FluidSim* sim)
{
    if (sim == NULL) {
        return;
    }

    free(sim->mass);
    free(sim->next_mass);
    free(sim->solid);

    sim->mass = NULL;
    sim->next_mass = NULL;
    sim->solid = NULL;

    sim->width = 0;
    sim->height = 0;
}

int fluid_sim_index(int x, int y)
{
    return y * SIM_WIDTH + x;
}

int fluid_sim_in_bounds(int x, int y)
{
    return (x >= 0 && x < SIM_WIDTH && y >= 0 && y < SIM_HEIGHT);
}

void fluid_sim_clear(FluidSim* sim)
{
    int i;

    if (sim == NULL) {
        return;
    }

    for (i = 0; i < SIM_CELL_COUNT; ++i) {
        sim->mass[i] = 0.0f;
        sim->next_mass[i] = 0.0f;
        sim->solid[i] = CELL_EMPTY;
    }
}

void fluid_sim_clear_mass(FluidSim* sim)
{
    int i;

    if (sim == NULL) {
        return;
    }

    for (i = 0; i < SIM_CELL_COUNT; ++i) {
        sim->mass[i] = 0.0f;
        sim->next_mass[i] = 0.0f;
    }
}

void fluid_sim_clear_solids(FluidSim* sim)
{
    int i;

    if (sim == NULL) {
        return;
    }

    for (i = 0; i < SIM_CELL_COUNT; ++i) {
        sim->solid[i] = CELL_EMPTY;
    }
}

void fluid_sim_set_solid(FluidSim* sim, int x, int y, unsigned char value)
{
    int idx;

    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return;
    }

    idx = fluid_sim_index(x, y);
    sim->solid[idx] = value;
}

unsigned char fluid_sim_get_solid(const FluidSim* sim, int x, int y)
{
    int idx;

    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return CELL_SOLID;
    }

    idx = fluid_sim_index(x, y);
    return sim->solid[idx];
}

void fluid_sim_set_mass(FluidSim* sim, int x, int y, float value)
{
    int idx;

    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return;
    }

    if (value < 0.0f) {
        value = 0.0f;
    }

    idx = fluid_sim_index(x, y);
    sim->mass[idx] = value;
}

float fluid_sim_get_mass(const FluidSim* sim, int x, int y)
{
    int idx;

    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return 0.0f;
    }

    idx = fluid_sim_index(x, y);
    return sim->mass[idx];
}

void fluid_sim_add_border_walls(FluidSim* sim)
{
    int x;
    int y;

    if (sim == NULL) {
        return;
    }

    for (x = 0; x < SIM_WIDTH; ++x) {
        fluid_sim_set_solid(sim, x, 0, CELL_SOLID);
        fluid_sim_set_solid(sim, x, SIM_HEIGHT - 1, CELL_SOLID);
    }

    for (y = 0; y < SIM_HEIGHT; ++y) {
        fluid_sim_set_solid(sim, 0, y, CELL_SOLID);
        fluid_sim_set_solid(sim, SIM_WIDTH - 1, y, CELL_SOLID);
    }
}

void fluid_sim_add_rect_solid(FluidSim* sim, int start_x, int start_y, int width, int height)
{
    int x;
    int y;
    int end_x;
    int end_y;

    if (sim == NULL || width <= 0 || height <= 0) {
        return;
    }

    end_x = start_x + width;
    end_y = start_y + height;

    for (y = start_y; y < end_y; ++y) {
        for (x = start_x; x < end_x; ++x) {
            if (fluid_sim_in_bounds(x, y)) {
                fluid_sim_set_solid(sim, x, y, CELL_SOLID);
            }
        }
    }
}

void fluid_sim_add_rect_mass(FluidSim* sim, int start_x, int start_y, int width, int height, float value)
{
    int x;
    int y;
    int end_x;
    int end_y;

    if (sim == NULL || width <= 0 || height <= 0) {
        return;
    }

    if (value < 0.0f) {
        value = 0.0f;
    }

    end_x = start_x + width;
    end_y = start_y + height;

    for (y = start_y; y < end_y; ++y) {
        for (x = start_x; x < end_x; ++x) {
            if (fluid_sim_in_bounds(x, y) && fluid_sim_get_solid(sim, x, y) != CELL_SOLID) {
                fluid_sim_set_mass(sim, x, y, value);
            }
        }
    }
}

void fluid_sim_swap_buffers(FluidSim* sim)
{
    float* temp;

    if (sim == NULL) {
        return;
    }

    temp = sim->mass;
    sim->mass = sim->next_mass;
    sim->next_mass = temp;
}

void fluid_sim_print_mass(const FluidSim* sim)
{
    int x;
    int y;

    if (sim == NULL) {
        return;
    }

    for (y = 0; y < SIM_HEIGHT; ++y) {
        for (x = 0; x < SIM_WIDTH; ++x) {
            printf("%4.1f ", fluid_sim_get_mass(sim, x, y));
        }
        printf("\n");
    }
}

void fluid_sim_print_solid(const FluidSim* sim)
{
    int x;
    int y;

    if (sim == NULL) {
        return;
    }

    for (y = 0; y < SIM_HEIGHT; ++y) {
        for (x = 0; x < SIM_WIDTH; ++x) {
            if (fluid_sim_get_solid(sim, x, y) == CELL_SOLID) {
                printf("#");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
}

void fluid_sim_print_combined(const FluidSim* sim)
{
    int x;
    int y;
    float mass;

    if (sim == NULL) {
        return;
    }

    for (y = 0; y < SIM_HEIGHT; ++y) {
        for (x = 0; x < SIM_WIDTH; ++x) {
            if (fluid_sim_get_solid(sim, x, y) == CELL_SOLID) {
                printf("#");
                continue;
            }

            mass = fluid_sim_get_mass(sim, x, y);

            if (mass <= MIN_MASS) {
                printf(".");
            } else if (mass < 0.25f) {
                printf("-");
            } else if (mass < 0.50f) {
                printf("~");
            } else if (mass < 0.90f) {
                printf("W");
            } else {
                printf("@");
            }
        }
        printf("\n");
    }
}