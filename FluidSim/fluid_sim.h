#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <stddef.h>

#define SIM_WIDTH 64
#define SIM_HEIGHT 64
#define SIM_CELL_COUNT (SIM_WIDTH * SIM_HEIGHT)

#define MAX_MASS 1.0f
#define MIN_MASS 0.0001f

#define DOWN_RATE 0.90f
#define DIAG_RATE 0.2f
#define SIDE_RATE 0.0005f

#define CELL_EMPTY 0
#define CELL_SOLID 1

typedef struct FluidSim {
    int width;
    int height;

    float* mass;
    float* next_mass;
    unsigned char* solid;
} FluidSim;

int fluid_sim_init(FluidSim* sim);
void fluid_sim_free(FluidSim* sim);

int fluid_sim_index(int x, int y);
int fluid_sim_in_bounds(int x, int y);

void fluid_sim_clear(FluidSim* sim);
void fluid_sim_clear_mass(FluidSim* sim);
void fluid_sim_clear_solids(FluidSim* sim);

void fluid_sim_set_solid(FluidSim* sim, int x, int y, unsigned char value);
unsigned char fluid_sim_get_solid(const FluidSim* sim, int x, int y);

void fluid_sim_set_mass(FluidSim* sim, int x, int y, float value);
float fluid_sim_get_mass(const FluidSim* sim, int x, int y);

void fluid_sim_add_border_walls(FluidSim* sim);
void fluid_sim_add_rect_solid(FluidSim* sim, int start_x, int start_y, int width, int height);
void fluid_sim_add_rect_mass(FluidSim* sim, int start_x, int start_y, int width, int height, float value);

void fluid_sim_swap_buffers(FluidSim* sim);

void fluid_sim_print_mass(const FluidSim* sim);
void fluid_sim_print_solid(const FluidSim* sim);
void fluid_sim_print_combined(const FluidSim* sim);

#endif