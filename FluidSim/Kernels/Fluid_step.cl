inline int fluid_index(int x, int y, int width)
{
    return y * width + x;
}

inline int fluid_in_bounds(int x, int y, int width, int height)
{
    return (x >= 0 && x < width && y >= 0 && y < height);
}

inline int fluid_is_open_cell(__global const uchar* solid, int x, int y, int width, int height)
{
    int idx;

    if (!fluid_in_bounds(x, y, width, height)) {
        return 0;
    }

    idx = fluid_index(x, y, width);
    return solid[idx] == 0;
}

inline float fluid_mass_at(
    __global const float* mass,
    __global const uchar* solid,
    int x,
    int y,
    int width,
    int height
)
{
    int idx;

    if (!fluid_in_bounds(x, y, width, height)) {
        return 0.0f;
    }

    idx = fluid_index(x, y, width);

    if (solid[idx] != 0) {
        return 0.0f;
    }

    return mass[idx];
}

inline float positive_min(float a, float b)
{
    float m = fmin(a, b);
    return fmax(m, 0.0f);
}

__kernel void fluid_step(
    __global const float* mass,
    __global float* next_mass,
    __global const uchar* solid,
    const int width,
    const int height,
    const float max_mass,
    const float min_mass,
    const float down_rate,
    const float diag_rate,
    const float side_rate
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx;

    float self_mass;
    float new_mass_value;

    float up_mass;
    float down_mass;
    float left_mass;
    float right_mass;
    float up_left_mass;
    float up_right_mass;
    float down_left_mass;
    float down_right_mass;

    float flow_in;
    float flow_out;
    float diff;
    float capacity;

    if (x >= width || y >= height) {
        return;
    }

    idx = fluid_index(x, y, width);

    if (solid[idx] != 0) {
        next_mass[idx] = 0.0f;
        return;
    }

    self_mass = mass[idx];
    new_mass_value = self_mass;

    up_mass = fluid_mass_at(mass, solid, x, y - 1, width, height);
    down_mass = fluid_mass_at(mass, solid, x, y + 1, width, height);
    left_mass = fluid_mass_at(mass, solid, x - 1, y, width, height);
    right_mass = fluid_mass_at(mass, solid, x + 1, y, width, height);
    up_left_mass = fluid_mass_at(mass, solid, x - 1, y - 1, width, height);
    up_right_mass = fluid_mass_at(mass, solid, x + 1, y - 1, width, height);
    down_left_mass = fluid_mass_at(mass, solid, x - 1, y + 1, width, height);
    down_right_mass = fluid_mass_at(mass, solid, x + 1, y + 1, width, height);

    /* --- Inflow from upper neighbors --- */

    if (fluid_is_open_cell(solid, x, y - 1, width, height)) {
        diff = up_mass - self_mass;
        capacity = max_mass - self_mass;
        flow_in = positive_min(diff * down_rate, capacity);
        new_mass_value += flow_in;
    }

    if (fluid_is_open_cell(solid, x - 1, y - 1, width, height)) {
        diff = up_left_mass - self_mass;
        capacity = max_mass - self_mass;
        flow_in = positive_min(diff * diag_rate, capacity);
        new_mass_value += flow_in;
    }

    if (fluid_is_open_cell(solid, x + 1, y - 1, width, height)) {
        diff = up_right_mass - self_mass;
        capacity = max_mass - self_mass;
        flow_in = positive_min(diff * diag_rate, capacity);
        new_mass_value += flow_in;
    }

    /* --- Side inflow for local equalization --- */

    if (fluid_is_open_cell(solid, x - 1, y, width, height)) {
        diff = left_mass - self_mass;
        capacity = max_mass - self_mass;
        flow_in = positive_min(diff * side_rate, capacity);
        new_mass_value += flow_in;
    }

    if (fluid_is_open_cell(solid, x + 1, y, width, height)) {
        diff = right_mass - self_mass;
        capacity = max_mass - self_mass;
        flow_in = positive_min(diff * side_rate, capacity);
        new_mass_value += flow_in;
    }

    /* --- Outflow to lower neighbors --- */

    if (fluid_is_open_cell(solid, x, y + 1, width, height)) {
        diff = self_mass - down_mass;
        flow_out = positive_min(diff * down_rate, self_mass);
        new_mass_value -= flow_out;
    }

    if (fluid_is_open_cell(solid, x - 1, y + 1, width, height)) {
        diff = self_mass - down_left_mass;
        flow_out = positive_min(diff * diag_rate, self_mass);
        new_mass_value -= flow_out;
    }

    if (fluid_is_open_cell(solid, x + 1, y + 1, width, height)) {
        diff = self_mass - down_right_mass;
        flow_out = positive_min(diff * diag_rate, self_mass);
        new_mass_value -= flow_out;
    }

    /* --- Side outflow --- */

    if (fluid_is_open_cell(solid, x - 1, y, width, height)) {
        diff = self_mass - left_mass;
        flow_out = positive_min(diff * side_rate, self_mass);
        new_mass_value -= flow_out;
    }

    if (fluid_is_open_cell(solid, x + 1, y, width, height)) {
        diff = self_mass - right_mass;
        flow_out = positive_min(diff * side_rate, self_mass);
        new_mass_value -= flow_out;
    }

    if (new_mass_value < min_mass) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value < 0.0f) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value > max_mass) {
        new_mass_value = max_mass;
    }

    next_mass[idx] = new_mass_value;
}