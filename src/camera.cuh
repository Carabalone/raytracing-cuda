#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include "material.cuh"
#include <curand_kernel.h>
#include <chrono>

class camera {
public:
    float  aspect_ratio = 1.0;  // Ratio of image width over height
    int    res_x        = 100;  // Rendered image width in pixel count
    int    spp          = 10;
    vec3 last_rand_vec3 = vec3(0,0,0);
    int    vfov         = 90;   //vertical fov in degs

    point3 center       = point3(0,0,0);
    vec3   lookat       = vec3(0,0,-1);
    vec3   up           = vec3(0,1,0);

    float defocus_angle = 0.0f;
    float focal_distance = 10.0f;

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
            pixel_color += ray_color(r, world, local_rand_state);
        }
        framebuffer[pixel_index] = pixel_color * pixel_samples_scale;
    }

    __host__ void initialize() {
        res_y = int(res_x / aspect_ratio);
        res_y = (res_y < 1) ? 1 : res_y;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        float viewport_height = 2.0f * h * focal_distance;
        float viewport_width = viewport_height * (double(res_x)/res_y);

        // camera basis vectors
        w = normalize(center - lookat); // -forward
        u = normalize(cross(up, w));    // right
        v = cross(w, u);                // up

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = viewport_width * u;
        auto viewport_v = viewport_height * -v;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / res_x;
        pixel_delta_v = viewport_v / res_y;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - (focal_distance * w) - viewport_u/2.0f - viewport_v/2.0f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        float defocus_radius = focal_distance * tan(degrees_to_radians(defocus_angle/2.0f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

        pixel_samples_scale = 1.0f / spp;
    }

    __host__ int get_res_y() {
        return res_y;
    }


private:
    int    res_y;          // Rendered image height
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    float  pixel_samples_scale;
    int max_depth = 10;
    vec3   u, v, w;        // basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    __device__ color ray_color(const ray& r, hittable** world, curandState &rand_state) const {
        hit_record rec;
        ray current_ray = r;
        // float attenuation = 0.5f;
        // float current_attenuation = 1.0f;
        int count = 0;
        ray scattered = r;
        color accumulated_color = color(1, 1, 1);
        color current_attenuation = color(1, 1, 1);

        while (count < max_depth) {
            if ((*world)->hit(current_ray, interval(0.001f, infinity), rec)) {

                // printf("rec.mat: %p\n", rec.mat);
                if (rec.mat->scatter(current_ray, rec, current_attenuation, scattered, rand_state)) {
                    current_ray = scattered;
                    accumulated_color = accumulated_color * current_attenuation;
                } else {
                    return color(0,0,0);
                }
            } else { // background color
                vec3 unit_direction = normalize(r.direction());
                auto a = 0.5f * (unit_direction.y() + 1.0f);
                color background = ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
                return accumulated_color * background;
            }

            count++;
        }

        return color(0,0,0);
    }

    __device__ vec3 sample_square(curandState &rand_state) const {

        return vec3(
            curand_uniform(&rand_state) - 0.5f,
            curand_uniform(&rand_state) - 0.5f,
            0.0f
        );
    }

    __device__ point3 defocus_disk_sample(curandState &rand_state) const {
        // Returns a random point in the camera defocus disk.
        point3 p = random_in_unit_disk(rand_state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    __device__ ray get_ray(int i, int j, curandState &rand_state) {
        auto offset = sample_square(rand_state);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0.0f) ? center : defocus_disk_sample(rand_state);
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction); 
    }
};

#endif
