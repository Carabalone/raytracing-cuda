#ifndef UTILITY_H
#define UTILITY_H
#include <curand_kernel.h>

// returns from [0, 1[
__device__ float inline gpu_rand(curandState &state) {
    return curand_uniform(&state);
}

__device__ float inline gpu_rand(curandState &state, float min, float max) {
    return min + (max - min) * curand_uniform(&state);
}

#endif // UTILITY_H
