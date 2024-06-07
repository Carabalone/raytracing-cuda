#include "rtweekend.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.cuh"
#include "clock.h"

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
void render(camera cam, vec3* framebuffer, hittable** world) {
    cam.render(framebuffer, world);
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
    // Camera
    camera cam;
    cam.aspect_ratio = 16.0f/9.0f;
    cam.res_x = 400;
    cam.initialize();
    auto res_y = cam.get_res_y();

    // Framebuffer
    int num_pixels = cam.res_x * res_y;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *framebuffer;

    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));

    int threads_x = 8, threads_y = 8;
    dim3 threads(threads_x, threads_y);
    dim3 blocks(cam.res_x / threads_x  + 1, res_y / threads_y + 1);

    // world creation
    hittable **d_list, **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    create_world<<<1,1>>>(d_list,d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    rtweekend::clock c;

    c.start();
    render<<<blocks, threads>>>(cam, framebuffer, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    c.end();

    c.print();

    std::cout << "Saving framebuffer to file..." << std::endl;
    save_framebuffer_to_file(framebuffer, cam.res_x, res_y, "output/output.ppm");

    // freeing stuff
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(framebuffer));
    return 0;
}
