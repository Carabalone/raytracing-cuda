
#ifndef COLOR_H
#define COLOR_H

#include "vector.h"
#include "interval.cuh"

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color);
inline double linear_to_gamma(double linear_component);
#endif
