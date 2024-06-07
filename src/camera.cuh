#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "hittable.h"
#include <chrono>

class camera {
public:
    float  aspect_ratio = 1.0;  // Ratio of image width over height
    int    res_x        = 100;  // Rendered image width in pixel count

    // __host__ write_framebuffer_to_file(vec3* framebuffer) {
    //     std::ofstream file(output_filename);
    //     std::streambuf* coutBuffer = std::cout.rdbuf();
    //     std::cout.rdbuf(file.rdbuf());
    //
    //     std::cout << "P3\n" << res_x << " " << res_y << "\n255\n";
    //     for (int j = 0; j < res_y; j++) {
    //         for (int i = 0; i < res_x; i++) {
    //             size_t pixel_index = j*res_x + i;
    //             vec3 rgb = framebuffer[pixel_index];
    //
    //             write_color(std::cout, rgb);
    //         }
    //     }
    //
    //     std::cout.rdbuf(coutBuffer);
    //     std::cout << "Finished" << std::endl;
    // }
    //TODO: refactor framebuffer from vec3* to own class
    __device__ void render(vec3* framebuffer, hittable** world) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i >= res_x) || (j >= res_y)) return;

        int pixel_index = j*res_x + i;

        point3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        vec3 ray_direction = pixel_center - center;

        ray r(center, ray_direction);

        framebuffer[pixel_index] = ray_color(r, world);
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


    __device__ color ray_color(const ray& r, hittable** world) const {
        hit_record rec;
        if ((*world)->hit(r, interval(0, infinity), rec)) {
            return 0.5 * (rec.normal + color(1,1,1));
        }

        vec3 unit_direction = normalize(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
    }
};

#endif
