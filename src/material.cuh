#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
#include <curand_kernel.h>

class hit_record;

class material {
public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState& rand_state
    ) const {
        return false;
    }

};

class lambertian : public material {
private:
    color albedo;

public:
    __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
                 curandState& rand_state)
        const override {
        auto scatter_direction = rec.normal + random_unit_vector(rand_state);
        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material {
private:
    color albedo;
    float fuzz;
public:
    __device__ metal(const color& albedo, float _fuzz = 0.0f) : albedo(albedo), fuzz(_fuzz) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
                curandState& rand_state)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = normalize(reflected) + (fuzz * random_unit_vector(rand_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

};

#endif
