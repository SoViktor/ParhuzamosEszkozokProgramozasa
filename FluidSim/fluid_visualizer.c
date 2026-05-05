#include <stdio.h>
#include "fluid_visualizer.h"

static void write_pixel(FILE* f, unsigned char r, unsigned char g, unsigned char b)
{
    fputc(r, f);
    fputc(g, f);
    fputc(b, f);
}

int fluid_save_ppm(const FluidSim* sim, const char* filename, int scale)
{
    FILE* f;
    int x, y, sx, sy;
    int out_w, out_h;
    float m;
    unsigned char r, g, b;

    if (sim == NULL || filename == NULL || scale <= 0) {
        return 0;
    }

    f = fopen(filename, "wb");
    if (f == NULL) {
        return 0;
    }

    out_w = sim->width * scale;
    out_h = sim->height * scale;

    fprintf(f, "P6\n%d %d\n255\n", out_w, out_h);

    for (y = 0; y < sim->height; ++y) {
        for (sy = 0; sy < scale; ++sy) {
            for (x = 0; x < sim->width; ++x) {
                if (fluid_sim_get_solid(sim, x, y) == CELL_SOLID) {
                    r = 70; g = 70; b = 70;
                } else {
                    m = fluid_sim_get_mass(sim, x, y);

                    if (m <= MIN_MASS) {
                        r = 10; g = 10; b = 18;
                    } else {
                        if (m > 1.0f) m = 1.0f;
                        r = 20;
                        g = (unsigned char)(80 + 100 * m);
                        b = (unsigned char)(140 + 100 * m);
                    }
                }

                for (sx = 0; sx < scale; ++sx) {
                    write_pixel(f, r, g, b);
                }
            }
        }
    }

    fclose(f);
    return 1;
}