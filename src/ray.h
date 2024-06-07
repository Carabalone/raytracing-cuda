#ifndef RAY_H
#define RAY_H

#include <iostream>
#include "vector.h"

class ray {
private:
    point3 orig;
    vec3 dir;

public:
    __device__ ray(const point3 o, const vec3 d) : orig(o), dir(d) {}

    __device__ const point3& origin() const  { return orig; }
    __device__ const vec3& direction() const { return dir; }

    __device__ const point3 at(float t) const {
        return orig + dir * t;
    }
};

#endif // RAY_H
