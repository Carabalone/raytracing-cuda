#include "rtweekend.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
#include <chrono>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

__device__ color ray_color(const ray& r, hittable** world) {
    hit_record rec;
    if ((*world)->hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = normalize(r.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return ((1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f));
}

__global__
void render(vec3* framebuffer, int res_x, int res_y, vec3 pixel_delta_u, vec3 pixel_delta_v, point3 pixel00, point3 camera_center, hittable **d_world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= res_x) || (j >= res_y)) return;

    int pixel_index = j*res_x + i;

    point3 pixel_center = pixel00 + (i * pixel_delta_u) + (j * pixel_delta_v);
    vec3 ray_direction = pixel_center - camera_center;

    ray r(camera_center, ray_direction);

    framebuffer[pixel_index] = ray_color(r, d_world);
}

void save_framebuffer_to_file(vec3* framebuffer, int res_x, int res_y, std::string output_filename) {
    std::ofstream file(output_filename);
    std::streambuf* coutBuffer = std::cout.rdbuf();
    std::cout.rdbuf(file.rdbuf());

    std::cout << "P3\n" << res_x << " " << res_y << "\n255\n";
    for (int j = 0; j < res_y; j++) {
        for (int i = 0; i < res_x; i++) {
            size_t pixel_index = j*res_x + i;
            vec3 rgb = framebuffer[pixel_index];

            write_color(std::cout, rgb);
        }
    }

    std::cout.rdbuf(coutBuffer);
    std::cout << "Finished" << std::endl;
}

int main() {

    // Image setup
    float aspect_ratio = 16.0f / 9.0f;
    int res_x = 400;
    int res_y = int(res_x / aspect_ratio);
    res_y = (res_y < 1) ? 1 : res_y;
    int num_pixels = res_x * res_y;

    //Camera
    float focal_length = 1.0f;
    float viewport_height = 2.0f; // arbitrary
    float viewport_width = viewport_height * (float(res_x) / res_y); // real vs. approxiamate AR
    point3 camera_center = point3(0.0f, 0.0f, 0.0f);

    vec3 viewport_u = vec3(viewport_width, 0.0f, 0.0f);
    vec3 viewport_v = vec3(0.0f, -viewport_height, 0.0f);

    vec3 pixel_delta_u = viewport_u / res_x;
    vec3 pixel_delta_v = viewport_v / res_y;

    point3 viewport_upper_left = camera_center - vec3(0.0f, 0.0f, focal_length) - 
        viewport_u/2 - viewport_v/2;

    point3 pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v); // center of pixel 00


    // Framebuffer
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *framebuffer;

    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));

    int threads_x = 8, threads_y = 8;
    dim3 threads(threads_x, threads_y);
    dim3 blocks(res_x / threads_x  + 1, res_y / threads_y + 1);

    // world creation
    hittable **d_list, **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(framebuffer, res_x, res_y,
            pixel_delta_u, pixel_delta_v, pixel00_loc, camera_center, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    save_framebuffer_to_file(framebuffer, res_x, res_y, "output/output.ppm");

    checkCudaErrors(cudaFree(framebuffer));
    return 0;
}
