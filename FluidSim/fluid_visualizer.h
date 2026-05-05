#ifndef FLUID_VISUALIZER_H
#define FLUID_VISUALIZER_H

#include "fluid_sim.h"

int fluid_save_ppm(const FluidSim* sim, const char* filename, int scale);

#endif