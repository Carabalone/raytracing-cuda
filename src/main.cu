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
#include "material.cuh"
#include <curand_kernel.h>
#include "utility.cuh"
#include "material_manager.cuh"


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


__host__ void create_materials(material_manager* mat_manager) {
    printf("size of materials: \nlambertian: %lu\nmetal: %lu \ndieletric: %lu\n", sizeof(lambertian), sizeof(metal), sizeof(dieletric));
    material_info* ground = new material_info;
    ground->type = material_type::lambertian_t;
    ground->albedo = vec3(0.8, 0.8, 0.0);

    mat_manager->add_material(ground);

    material_info* mid = new material_info;
    mid->type = material_type::lambertian_t;
    mid->albedo = vec3(0.3, 0.3, 0.8);

    mat_manager->add_material(mid);

    material_info* right = new material_info;
    right->type   = material_type::metal_t;
    right->albedo = vec3(0.8, 0.6, 0.2);
    right->fuzz   = 0.2f;

    mat_manager->add_material(right);

    material_info* hollow_in = new material_info;
    hollow_in->type   = material_type::dieletric_t;
    hollow_in->refraction_index = 1.5f;

    mat_manager->add_material(hollow_in);

    material_info* hollow_out = new material_info;
    hollow_out->type   = material_type::dieletric_t;
    hollow_out->refraction_index = 1.0f / 1.5f;

    mat_manager->add_material(hollow_out);
}

// TODO: make create_materials again
__global__ void create_world(hittable **d_list, hittable **d_world, material_manager* mat_manager) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        material** mats = mat_manager->get_device_materials();

        d_list[0] = new sphere(vec3(0,0,-1.2f), 0.5,
                               mats[1]);
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                               mats[0]);
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                               mats[2]);
        d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
                               mats[3]);
        d_list[4] = new sphere(vec3(-1.0f, 0.0, -1.0), 0.4,
                               mats[4]);
        *d_world  = new hittable_list(d_list,5);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i=0; i < 5; i++) {
            delete ((sphere *)d_list[i])->mat; // TODO: delete this stuff in mat_manager
            delete d_list[i];
        }
        delete *d_world;
    }
}

__global__ void render_init(int seed, int res_x, int res_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= res_x) || (j >= res_y)) return;
    int pixel_index = j*res_x + i;

    curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

// don't need to copy cam to device explicitly, it's copied by value by cuda automatically
// and doesn't impact much given that it's just one kernel call.
__global__ void render(camera cam, vec3* framebuffer, hittable** world, curandState *rand_state) {
    cam.render(framebuffer, world, rand_state);
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

camera create_camera() {
    camera cam;
    cam.aspect_ratio = 16.0f/9.0f;
    cam.res_x  = 400;
    cam.spp    = 100;
    cam.center = point3(-2, 2, 1);
    cam.lookat = point3(0, 0, -1);
    cam.up     = point3(0, 1, 0);
    cam.vfov   = 30;

    cam.defocus_angle  = 0.1f;
    cam.focal_distance = 10.0f;
    cam.initialize();
    return cam;
}

int main() {
    // Camera
    camera cam = create_camera();
    int res_y  = cam.get_res_y();

    // materials: 
    material_manager* mat_manager;
    checkCudaErrors(cudaMallocManaged((void**)&mat_manager, sizeof(mat_manager)));
    create_materials(mat_manager);
    mat_manager->dispatch();

    create_device_materials<<<1, 1>>>(
        mat_manager->get_device_material_info(),
        mat_manager->get_device_materials(),
        mat_manager->size()
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Framebuffer
    int num_pixels = cam.res_x * res_y;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *framebuffer;

    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));

    int threads_x = 8, threads_y = 8;
    dim3 threads(threads_x, threads_y);
    dim3 blocks(cam.res_x / threads_x  + 1, res_y / threads_y + 1);

    // world creation
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 5*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));

    create_world<<<1,1>>>(d_list,d_world, mat_manager);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    curandState *rand_state;
    checkCudaErrors(cudaMalloc((void **)&rand_state, num_pixels * sizeof(curandState)));

    render_init<<<blocks, threads>>>(1984, cam.res_x, res_y, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    rtweekend::clock c;

    c.start();
    render<<<blocks, threads>>>(cam, framebuffer, d_world, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    c.end();
    c.print();

    std::cout << "Saving framebuffer to file..." << std::endl;
    save_framebuffer_to_file(framebuffer, cam.res_x, res_y, "output/output.ppm");

    // freeing stuff
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(framebuffer));
    return 0;
}


