#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include <curand_kernel.h>
#include <chrono>

class camera {
public:
    float  aspect_ratio = 1.0;  // Ratio of image width over height
    int    res_x        = 100;  // Rendered image width in pixel count
    int    spp          = 10;
    vec3 last_rand_vec3 = vec3(0,0,0);

    //TODO: refactor framebuffer from vec3* to own class
    __device__ void render(vec3* framebuffer, hittable** world, curandState *rand_state) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= res_x) || (j >= res_y)) return;

        int pixel_index = j*res_x + i;
        curandState local_rand_state = rand_state[pixel_index];
        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < spp; sample++) {
            ray r = get_ray(i, j, local_rand_state);
            pixel_color += ray_color(r, world);
        }
        framebuffer[pixel_index] = pixel_color * pixel_samples_scale;
    }

    __host__ void initialize() {
        res_y = int(res_x / aspect_ratio);
        res_y = (res_y < 1) ? 1 : res_y;

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        float focal_length = 1.0;
        float viewport_height = 2.0;
        float viewport_width = viewport_height * (double(res_x)/res_y);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / res_x;
        pixel_delta_v = viewport_v / res_y;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        pixel_samples_scale = 1.0f / spp;
    }

    __host__ int get_res_y() {
        return res_y;
    }


private:
    int    res_y;          // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    float  pixel_samples_scale;

    __device__ color ray_color(const ray& r, hittable** world) const {
        hit_record rec;
        if ((*world)->hit(r, interval(0, infinity), rec)) {
            return 0.5 * (rec.normal + color(1,1,1));
        }

        vec3 unit_direction = normalize(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
    }

    __device__ vec3 sample_square(curandState &rand_state) const {

        return vec3(
            curand_uniform(&rand_state) - 0.5f,
            curand_uniform(&rand_state) - 0.5f,
            0.0f
        );
    }

    __device__ ray get_ray(int i, int j, curandState &rand_state) {
        auto offset = sample_square(rand_state);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction); 
    }
};

#endif
