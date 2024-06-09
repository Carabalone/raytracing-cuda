#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
#include "utility.cuh"
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

class dieletric : public material {
public:
    __device__ dieletric(float _refraction_index) : refraction_index(_refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
                curandState& rand_state)
    const override {
        attenuation = color(1.0f);
        float ref_ind = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        vec3 unit_direction = normalize(r_in.direction());

        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        vec3 direction;

        // first predicate is equivalent to: if (sin(theta') > 1.0f)
        // i have no idea about why we get the reflectance > random_val 
        // ask peter sherley about it
        if (ref_ind * sin_theta > 1.0f || reflectance(cos_theta, ref_ind) > gpu_rand(rand_state)) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, ref_ind);
        }

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    float refraction_index;

    // schlick's approximation
    static __device__ float reflectance(float cosine, float _refraction_index) {
        float r0 = (1.0f - _refraction_index) / (1.0f + _refraction_index);
        r0 = r0 * r0;

        return r0 + (1.0f - r0)*pow((1.0f-cosine), 5);
    }
};

#endif
