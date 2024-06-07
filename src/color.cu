#include <iostream>
#include "vector.h"
#include "color.h"

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    int rbyte = int(255.999f * r);
    int gbyte = int(255.999f * g);
    int bbyte = int(255.999f * b);

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}
