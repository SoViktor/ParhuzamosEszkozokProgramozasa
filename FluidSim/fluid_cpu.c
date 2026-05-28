#include <math.h>

#include "fluid_cpu.h"

static float cpu_min_float(float a, float b)
{
    return (a < b) ? a : b;
}

static float cpu_max_float(float a, float b)
{
    return (a > b) ? a : b;
}

static float positive_min(float a, float b)
{
    float value = cpu_min_float(a, b);
    return cpu_max_float(value, 0.0f);
}

static float cell_capacity(int y, float max_mass)
{
    (void)y;
    return max_mass;
}

static int fluid_is_open_cpu(const FluidSim* sim, int x, int y)
{
    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return 0;
    }

    return fluid_sim_get_solid(sim, x, y) != CELL_SOLID;
}

static float fluid_get_mass_cpu(const FluidSim* sim, int x, int y)
{
    int idx;

    if (sim == NULL || !fluid_sim_in_bounds(x, y)) {
        return 0.0f;
    }

    idx = fluid_sim_index(x, y);

    if (sim->solid[idx] == CELL_SOLID) {
        return 0.0f;
    }

    return sim->mass[idx];
}

/*
    Visszateresi ertek:
    0 = nincs mozgas
    1 = le
    2 = bal-le
    3 = jobb-le
    4 = bal
    5 = jobb
*/
static int fluid_primary_direction_cpu(
    const FluidSim* sim,
    int sx,
    int sy,
    float max_mass,
    float min_mass
)
{
    float self_mass;
    float down_mass;
    float down_left_mass;
    float down_right_mass;
    float left_mass;
    float right_mass;
    float down_capacity;
    float down_left_capacity;
    float down_right_capacity;

    self_mass = fluid_get_mass_cpu(sim, sx, sy);

    if (self_mass <= min_mass) {
        return 0;
    }

    if (fluid_is_open_cpu(sim, sx, sy + 1)) {
        down_mass = fluid_get_mass_cpu(sim, sx, sy + 1);
        down_capacity = cell_capacity(sy + 1, max_mass);

        if (down_mass < down_capacity - min_mass && self_mass > down_mass + min_mass) {
            return 1;
        }
    }

    if (fluid_is_open_cpu(sim, sx - 1, sy + 1)) {
        down_left_mass = fluid_get_mass_cpu(sim, sx - 1, sy + 1);
        down_left_capacity = cell_capacity(sy + 1, max_mass);

        if (down_left_mass < down_left_capacity - min_mass && self_mass > down_left_mass + min_mass) {
            return 2;
        }
    }

    if (fluid_is_open_cpu(sim, sx + 1, sy + 1)) {
        down_right_mass = fluid_get_mass_cpu(sim, sx + 1, sy + 1);
        down_right_capacity = cell_capacity(sy + 1, max_mass);

        if (down_right_mass < down_right_capacity - min_mass && self_mass > down_right_mass + min_mass) {
            return 3;
        }
    }

    if (self_mass > max_mass + 0.50f) {
        if (fluid_is_open_cpu(sim, sx - 1, sy)) {
            left_mass = fluid_get_mass_cpu(sim, sx - 1, sy);

            if (self_mass > left_mass + min_mass) {
                return 4;
            }
        }

        if (fluid_is_open_cpu(sim, sx + 1, sy)) {
            right_mass = fluid_get_mass_cpu(sim, sx + 1, sy);

            if (self_mass > right_mass + min_mass) {
                return 5;
            }
        }
    }

    return 0;
}

static float compute_next_cell_cpu(const FluidSim* sim, int x, int y)
{
    int idx;
    int source_dir;
    float self_mass;
    float new_mass_value;
    float local_max;
    float source_mass;
    float target_mass;
    float flow_amount;
    float capacity;

    idx = fluid_sim_index(x, y);

    if (sim->solid[idx] == CELL_SOLID) {
        return 0.0f;
    }

    self_mass = sim->mass[idx];
    new_mass_value = self_mass;
    local_max = cell_capacity(y, MAX_MASS);

    if (fluid_is_open_cpu(sim, x, y - 1)) {
        source_dir = fluid_primary_direction_cpu(sim, x, y - 1, MAX_MASS, MIN_MASS);

        if (source_dir == 1) {
            source_mass = fluid_get_mass_cpu(sim, x, y - 1);
            target_mass = self_mass;
            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * DOWN_RATE, capacity);
            new_mass_value += flow_amount;
        }
    }

    if (fluid_is_open_cpu(sim, x - 1, y - 1)) {
        source_dir = fluid_primary_direction_cpu(sim, x - 1, y - 1, MAX_MASS, MIN_MASS);

        if (source_dir == 3) {
            source_mass = fluid_get_mass_cpu(sim, x - 1, y - 1);
            target_mass = self_mass;
            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * DIAG_RATE, capacity);
            new_mass_value += flow_amount;
        }
    }

    if (fluid_is_open_cpu(sim, x + 1, y - 1)) {
        source_dir = fluid_primary_direction_cpu(sim, x + 1, y - 1, MAX_MASS, MIN_MASS);

        if (source_dir == 2) {
            source_mass = fluid_get_mass_cpu(sim, x + 1, y - 1);
            target_mass = self_mass;
            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * DIAG_RATE, capacity);
            new_mass_value += flow_amount;
        }
    }

    if (fluid_is_open_cpu(sim, x - 1, y)) {
        source_dir = fluid_primary_direction_cpu(sim, x - 1, y, MAX_MASS, MIN_MASS);

        if (source_dir == 5) {
            source_mass = fluid_get_mass_cpu(sim, x - 1, y);
            target_mass = self_mass;
            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * SIDE_RATE, capacity);
            new_mass_value += flow_amount;
        }
    }

    if (fluid_is_open_cpu(sim, x + 1, y)) {
        source_dir = fluid_primary_direction_cpu(sim, x + 1, y, MAX_MASS, MIN_MASS);

        if (source_dir == 4) {
            source_mass = fluid_get_mass_cpu(sim, x + 1, y);
            target_mass = self_mass;
            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * SIDE_RATE, capacity);
            new_mass_value += flow_amount;
        }
    }

    source_dir = fluid_primary_direction_cpu(sim, x, y, MAX_MASS, MIN_MASS);

    if (source_dir == 1) {
        target_mass = fluid_get_mass_cpu(sim, x, y + 1);
        flow_amount = positive_min((self_mass - target_mass) * DOWN_RATE, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 2) {
        target_mass = fluid_get_mass_cpu(sim, x - 1, y + 1);
        flow_amount = positive_min((self_mass - target_mass) * DIAG_RATE, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 3) {
        target_mass = fluid_get_mass_cpu(sim, x + 1, y + 1);
        flow_amount = positive_min((self_mass - target_mass) * DIAG_RATE, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 4) {
        target_mass = fluid_get_mass_cpu(sim, x - 1, y);
        flow_amount = positive_min((self_mass - target_mass) * SIDE_RATE, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 5) {
        target_mass = fluid_get_mass_cpu(sim, x + 1, y);
        flow_amount = positive_min((self_mass - target_mass) * SIDE_RATE, self_mass);
        new_mass_value -= flow_amount;
    }

    if (new_mass_value < MIN_MASS) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value < 0.0f) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value > local_max) {
        new_mass_value = local_max;
    }

    return new_mass_value;
}

int fluid_cpu_step(FluidSim* sim)
{
    int x;
    int y;
    int idx;

    if (sim == NULL) {
        return 0;
    }

    for (y = 0; y < SIM_HEIGHT; ++y) {
        for (x = 0; x < SIM_WIDTH; ++x) {
            idx = fluid_sim_index(x, y);
            sim->next_mass[idx] = compute_next_cell_cpu(sim, x, y);
        }
    }

    fluid_sim_swap_buffers(sim);

    return 1;
}
