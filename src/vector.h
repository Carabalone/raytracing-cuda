#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <curand_kernel.h>
#include "utility.cuh"

class vec3 {

public:
    float coords[3];

    __host__ __device__ vec3() : coords{0.0f, 0.0f, 0.0f} {};
    __host__ __device__ vec3(float e0, float e1, float e2) : coords{e0, e1, e2} {};

    __host__ __device__ float x() const { return coords[0]; }
    __host__ __device__ float y() const { return coords[1]; }
    __host__ __device__ float z() const { return coords[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-coords[0], -coords[1], -coords[2]); }
    __host__ __device__ float operator[](int i) const { return coords[i]; }
    __host__ __device__ float& operator[](int i) { return coords[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        coords[0] += v.coords[0];
        coords[1] += v.coords[1];
        coords[2] += v.coords[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(float t) {
        coords[0] *= t;
        coords[1] *= t;
        coords[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(float t) {
        return *this *= 1/t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2];
    }

    // generates random vec with each component in [0.0f, 1.0f[
    // I am against generating this in 0-1 since it does not generate all possible vectors, but
    // ray tracing in one weekend does it and so will I.
    __device__ static inline vec3 random_gpu(curandState &rand_state) {
        return vec3(
            gpu_rand(rand_state),
            gpu_rand(rand_state),
            gpu_rand(rand_state)
        );
    }

    __device__ static inline vec3 random_gpu(curandState &rand_state, float min, float max) {
        return vec3(
            gpu_rand(rand_state, min, max),
            gpu_rand(rand_state, min, max),
            gpu_rand(rand_state, min, max)
        );
    }
};

using point3 = vec3;

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] + v.coords[0], u.coords[1] + v.coords[1], u.coords[2] + v.coords[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] - v.coords[0], u.coords[1] - v.coords[1], u.coords[2] - v.coords[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] * v.coords[0], u.coords[1] * v.coords[1], u.coords[2] * v.coords[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, float f) {
    return vec3(u.coords[0] * f, u.coords[1] * f, u.coords[2] * f);
}

__host__ __device__ inline vec3 operator*(float f, const vec3& u) {
    return u * f;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.coords[0] * v.coords[0]
         + u.coords[1] * v.coords[1]
         + u.coords[2] * v.coords[2];
}

__host__ __device__ inline bool operator==(const vec3& u, const vec3& v) {
    return (
        u.coords[0] == v.coords[0] &&
        u.coords[1] == v.coords[1] &&
        u.coords[2] == v.coords[2]
    );
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.coords[1] * v.coords[2] - u.coords[2] * v.coords[1],
                u.coords[2] * v.coords[0] - u.coords[0] * v.coords[2],
                u.coords[0] * v.coords[1] - u.coords[1] * v.coords[0]);
}

__host__ __device__ inline vec3 normalize(const vec3& v) {
    return v / v.length();
}


// TODO: extend these functions to work in the host as well.
// With macros, ex:
// #if defined(__CUDA_ARCH__)
//    gpu stuff
// #else 
//    cpu stuff
// #endif
__device__ static inline vec3 random_in_unit_sphere(curandState &rand_state) {
    while (true) {
        auto p = vec3::random_gpu(rand_state, -1.0f, 1.0f);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

__device__ inline vec3 random_unit_vector(curandState &rand_state) {
    return normalize(random_in_unit_sphere(rand_state));
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState &rand_state) {
    vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif // VEC3_H
