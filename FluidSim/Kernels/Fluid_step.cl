inline int fluid_index(int x, int y, int width)
{
    return y * width + x;
}

inline int fluid_in_bounds(int x, int y, int width, int height)
{
    return (x >= 0 && x < width && y >= 0 && y < height);
}

inline int fluid_is_open(__global const uchar* solid, int x, int y, int width, int height)
{
    int idx;

    if (!fluid_in_bounds(x, y, width, height)) {
        return 0;
    }

    idx = fluid_index(x, y, width);
    return (solid[idx] == 0);
}

inline float fluid_get_mass(
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
    float value = fmin(a, b);
    return fmax(value, 0.0f);
}

inline float cell_capacity(int y, float max_mass)
{
    return max_mass;
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
inline int fluid_primary_direction(
    __global const float* mass,
    __global const uchar* solid,
    int sx,
    int sy,
    int width,
    int height,
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

    self_mass = fluid_get_mass(mass, solid, sx, sy, width, height);

    if (self_mass <= min_mass) {
        return 0;
    }

    /*
        1. Lefele:
        akkor mehet lefele, ha az also cella nyitott
        es meg van benne kapacitas.
    */
    if (fluid_is_open(solid, sx, sy + 1, width, height)) {
        down_mass = fluid_get_mass(mass, solid, sx, sy + 1, width, height);
        down_capacity = cell_capacity(sy + 1, max_mass);

        if (down_mass < down_capacity - min_mass && self_mass > down_mass + min_mass) {
            return 1;
        }
    }

    /*
        2. Atlosan balra-le:
        csak akkor, ha egyenesen lefele nem tudott.
    */
    if (fluid_is_open(solid, sx - 1, sy + 1, width, height)) {
        down_left_mass = fluid_get_mass(mass, solid, sx - 1, sy + 1, width, height);
        down_left_capacity = cell_capacity(sy + 1, max_mass);

        if (down_left_mass < down_left_capacity - min_mass && self_mass > down_left_mass + min_mass) {
            return 2;
        }
    }

    /*
        3. Atlosan jobbra-le:
        csak akkor, ha lefele es balra-le sem ment.
    */
    if (fluid_is_open(solid, sx + 1, sy + 1, width, height)) {
        down_right_mass = fluid_get_mass(mass, solid, sx + 1, sy + 1, width, height);
        down_right_capacity = cell_capacity(sy + 1, max_mass);

        if (down_right_mass < down_right_capacity - min_mass && self_mass > down_right_mass + min_mass) {
            return 3;
        }
    }

    /*
        4. Oldalra:
        csak akkor, ha mar tultoltott.
        Ez gatolja meg, hogy mindenhol ritka folyadek legyen.
    */
    if (self_mass > max_mass + 0.50f) {
        if (fluid_is_open(solid, sx - 1, sy, width, height)) {
            left_mass = fluid_get_mass(mass, solid, sx - 1, sy, width, height);

            if (self_mass > left_mass + min_mass) {
                return 4;
            }
        }

        if (fluid_is_open(solid, sx + 1, sy, width, height)) {
            right_mass = fluid_get_mass(mass, solid, sx + 1, sy, width, height);

            if (self_mass > right_mass + min_mass) {
                return 5;
            }
        }
    }

    return 0;
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
    float local_max;

    int source_dir;

    float source_mass;
    float target_mass;
    float flow_amount;
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

    local_max = cell_capacity(y, max_mass);

    /*
        Bejovo folyadek felulrol:
        a fenti cella csak akkor ad ide, ha neki ez az elso valasztott iranya.
    */
    if (fluid_is_open(solid, x, y - 1, width, height)) {
        source_dir = fluid_primary_direction(
            mass, solid,
            x, y - 1,
            width, height,
            max_mass,
            min_mass
        );

        if (source_dir == 1) {
            source_mass = fluid_get_mass(mass, solid, x, y - 1, width, height);
            target_mass = self_mass;

            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * down_rate, capacity);

            new_mass_value += flow_amount;
        }
    }

    /*
        Bejovo folyadek bal-fentrol:
        akkor johet, ha a bal-felso cella jobbra-le akar menni.
    */
    if (fluid_is_open(solid, x - 1, y - 1, width, height)) {
        source_dir = fluid_primary_direction(
            mass, solid,
            x - 1, y - 1,
            width, height,
            max_mass,
            min_mass
        );

        if (source_dir == 3) {
            source_mass = fluid_get_mass(mass, solid, x - 1, y - 1, width, height);
            target_mass = self_mass;

            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * diag_rate, capacity);

            new_mass_value += flow_amount;
        }
    }

    /*
        Bejovo folyadek jobb-fentrol:
        akkor johet, ha a jobb-felso cella balra-le akar menni.
    */
    if (fluid_is_open(solid, x + 1, y - 1, width, height)) {
        source_dir = fluid_primary_direction(
            mass, solid,
            x + 1, y - 1,
            width, height,
            max_mass,
            min_mass
        );

        if (source_dir == 2) {
            source_mass = fluid_get_mass(mass, solid, x + 1, y - 1, width, height);
            target_mass = self_mass;

            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * diag_rate, capacity);

            new_mass_value += flow_amount;
        }
    }

    /*
        Bejovo folyadek balrol:
        csak akkor, ha a bal cella oldalra, jobbra akar menni.
    */
    if (fluid_is_open(solid, x - 1, y, width, height)) {
        source_dir = fluid_primary_direction(
            mass, solid,
            x - 1, y,
            width, height,
            max_mass,
            min_mass
        );

        if (source_dir == 5) {
            source_mass = fluid_get_mass(mass, solid, x - 1, y, width, height);
            target_mass = self_mass;

            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * side_rate, capacity);

            new_mass_value += flow_amount;
        }
    }

    /*
        Bejovo folyadek jobbrol:
        csak akkor, ha a jobb cella oldalra, balra akar menni.
    */
    if (fluid_is_open(solid, x + 1, y, width, height)) {
        source_dir = fluid_primary_direction(
            mass, solid,
            x + 1, y,
            width, height,
            max_mass,
            min_mass
        );

        if (source_dir == 4) {
            source_mass = fluid_get_mass(mass, solid, x + 1, y, width, height);
            target_mass = self_mass;

            capacity = local_max - new_mass_value;
            flow_amount = positive_min((source_mass - target_mass) * side_rate, capacity);

            new_mass_value += flow_amount;
        }
    }

    /*
        Kiaramlas:
        a sajat cella csak egyetlen, prioritas szerint elso iranyba ad le.
    */
    source_dir = fluid_primary_direction(
        mass, solid,
        x, y,
        width, height,
        max_mass,
        min_mass
    );

    if (source_dir == 1) {
        target_mass = fluid_get_mass(mass, solid, x, y + 1, width, height);
        flow_amount = positive_min((self_mass - target_mass) * down_rate, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 2) {
        target_mass = fluid_get_mass(mass, solid, x - 1, y + 1, width, height);
        flow_amount = positive_min((self_mass - target_mass) * diag_rate, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 3) {
        target_mass = fluid_get_mass(mass, solid, x + 1, y + 1, width, height);
        flow_amount = positive_min((self_mass - target_mass) * diag_rate, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 4) {
        target_mass = fluid_get_mass(mass, solid, x - 1, y, width, height);
        flow_amount = positive_min((self_mass - target_mass) * side_rate, self_mass);
        new_mass_value -= flow_amount;
    }
    else if (source_dir == 5) {
        target_mass = fluid_get_mass(mass, solid, x + 1, y, width, height);
        flow_amount = positive_min((self_mass - target_mass) * side_rate, self_mass);
        new_mass_value -= flow_amount;
    }

    if (new_mass_value < min_mass) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value < 0.0f) {
        new_mass_value = 0.0f;
    }

    if (new_mass_value > local_max) {
        new_mass_value = local_max;
    }

    next_mass[idx] = new_mass_value;
}