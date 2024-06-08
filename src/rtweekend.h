#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <memory>
#include "constants.h"


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Common Headers

#include "vector.h"
#include "color.h"
#include "ray.h"
#include "interval.cuh"

#endif
