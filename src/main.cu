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
#include "color.h"
#include "material_manager.cuh"
#include "geometry_manager.cuh"

__host__ void create_materials(material_manager* mat_manager) {
    printf("size of materials: \nlambertian: %lu\nmetal: %lu \ndieletric: %lu\n", sizeof(lambertian), sizeof(metal), sizeof(dieletric));

    // random 10 lambertian materials
    for (int i=0; i < 10; i++) {
        material_info* mat = new material_info;
        mat->type = material_type::lambertian_t;
        mat->albedo = random_color();
        mat_manager->add_material(mat);
    }

    // random 3 metal materials
    for (int i=0; i < 3; i++) {
        material_info* mat = new material_info;
        mat->type = material_type::metal_t;
        mat->albedo = random_color();
        mat->fuzz = float(random_double());
        mat_manager->add_material(mat);
    }

    // random 3 dieletric materials
    for (int i=0; i < 3; i++) {
        material_info* mat = new material_info;
        mat->type = material_type::dieletric_t;
        mat->refraction_index = float(random_double());
        mat_manager->add_material(mat);
    }

}

// TODO: make create_materials again
__global__ void create_world(hittable **d_list, hittable **d_world, material_manager* mat_manager, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = rand_state[0];
        material** mats = mat_manager->get_device_materials();

        // for (int a=-11; a < 11; a++) {
        //     for (int b=-11; b < 11; b++) {
        //         auto choose_mat_type = gpu_rand(local_rand_state);
        //         point3 center(a + 0.9 * gpu_rand(local_rand_state), 0.2, b + 0.9 * gpu_rand(local_rand_state));
        //
        //         if ((center - point3(4, 0.2, 0)).length() > 0.9f) {
        //             printf("acess: %d/%d\n", ((a+11) * (b+11) + (b+11)), (22*22));
        //             if (choose_mat_type < 0.8) {
        //                 auto choose_mat = int(gpu_rand(local_rand_state) * 10.5f);
        //                 printf("choose_mat: %d/%d\n", choose_mat, mat_manager->size());
        //                 d_list[(a + 11) * (b + 11) + b + 11] = new sphere(center, 0.2, mats[choose_mat]);
        //             } else if (choose_mat_type < 0.95) {
        //                 auto choose_mat = 11 + int(gpu_rand(local_rand_state) * 3);
        //                 printf("choose_mat: %d/%d\n", choose_mat, mat_manager->size());
        //                 d_list[(a + 11) * (b + 11) + b + 11] = new sphere(center, 0.2, mats[choose_mat]);
        //             } else {
        //                 auto choose_mat = 13 + int(gpu_rand(local_rand_state) * 3);
        //                 printf("choose_mat: %d/%d\n", choose_mat, mat_manager->size());
        //                 d_list[(a + 11) * (b + 11) + b + 11] = new sphere(center, 0.2, mats[choose_mat]);
        //             } 
        //         }
        //     }
        // }

        // d_list[0] = new sphere(vec3(0,0,-1.2f), 0.5,
        //                        mats[1]);
        // d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
        //                        mats[0]);
        // d_list[2] = new sphere(vec3(1,0,-1), 0.5,
        //                        mats[2]);
        // d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
        //                        mats[3]);
        // d_list[4] = new sphere(vec3(-1.0f, 0.0, -1.0), 0.4,
        //                        mats[4]);
        // *d_world  = new hittable_list(d_list, 5);
    }
}

__host__ void create_geometries(geometry_manager* geom_manager, material** mats) {


    // Geometry setup
    geometry_info* geom0 = new geometry_info;
    geom0->type = sphere_t;
    geom0->center = point3(0, 0, -1.2f);
    geom0->radius = 0.5f;
    geom0->mat = mats[11];

    geometry_info* geom1 = new geometry_info;
    geom1->type = sphere_t;
    geom1->center = point3(0, -100.5, -1);
    geom1->radius = 100;
    geom1->mat = mats[0];

    geometry_info* geom2 = new geometry_info;
    geom2->type = sphere_t;
    geom2->center = point3(1, 0, -1);
    geom2->radius = 0.5f;
    geom2->mat = mats[2];

    geometry_info* geom3 = new geometry_info;
    geom3->type = sphere_t;
    geom3->center = point3(-1, 0, -1);
    geom3->radius = 0.5f;
    geom3->mat = mats[3];

    geometry_info* geom4 = new geometry_info;
    geom4->type = sphere_t;
    geom4->center = point3(-1.0f, 0.0, -1.0);
    geom4->radius = 0.4f;
    geom4->mat = mats[4];

    // Add geometries to the geometry manager
    geom_manager->add_geometry(geom0);
    geom_manager->add_geometry(geom1);
    geom_manager->add_geometry(geom2);
    geom_manager->add_geometry(geom3);
    geom_manager->add_geometry(geom4);

    std::cout << "Before Dispatching to device" << std::endl;

    // Dispatch geometries to device memory
    geom_manager->dispatch();

    std::cout << "After Dispatching to device" << std::endl;

    // Create device geometries
    geometry_info** d_geometry_info = geom_manager->get_device_geometry_info();
    hittable** d_geometries = geom_manager->get_device_geometries();
    hittable** d_world = geom_manager->get_world();
    hittable_list* hl = geom_manager->get_hlist();
    int num_geometries = geom_manager->size();

    std::cout << "Creating device geometries: " << std::endl;

    create_device_geometries<<<1, 1>>>(d_geometry_info, d_geometries, hl, d_world, num_geometries);

    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Created device geometries\n";
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // for(int i=0; i < 16; i++) {
        //     delete ((sphere *)d_list[i])->mat; // TODO: delete this stuff in mat_manager
        //     delete d_list[i];
        // }
        // delete *d_world;
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
    cam.spp    = 25;
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

    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));
    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n", deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n", deviceProp.totalGlobalMem/1048576.0);
    printf("        free memory: %.1f MiB\n", gpu_free_mem/1048576.0);
    printf("\n");


    // Camera
    camera cam = create_camera();
    int res_y  = cam.get_res_y();

    // Framebuffer
    int num_pixels = cam.res_x * res_y;
    size_t framebuffer_size = num_pixels * sizeof(vec3);
    vec3 *framebuffer;

    checkCudaErrors(cudaMallocManaged((void **)&framebuffer, framebuffer_size));

    // materials: 
    material_manager* mat_manager;
    checkCudaErrors(cudaMallocManaged((void**)&mat_manager, sizeof(mat_manager)));
    create_materials(mat_manager);
    mat_manager->dispatch();

    int threads_x = 8, threads_y = 8;
    dim3 threads(threads_x, threads_y);
    dim3 blocks(cam.res_x / threads_x  + 1, res_y / threads_y + 1);

    curandState *rand_state;
    checkCudaErrors(cudaMalloc((void **)&rand_state, num_pixels * sizeof(curandState)));

    render_init<<<blocks, threads>>>(1984, cam.res_x, res_y, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_device_materials<<<1, 1>>>(
        mat_manager->get_device_material_info(),
        mat_manager->get_device_materials(),
        mat_manager->size()
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // world creation
    // hittable **d_list;
    // checkCudaErrors(cudaMalloc((void **)&d_list, 22*22*sizeof(hittable *)));
    // hittable **d_world;
    // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    //
    // create_world<<<1,1>>>(d_list,d_world, mat_manager, rand_state);

    geometry_manager* geom_manager;
    std::cout << "Reached geom manager;" << std::endl;
    checkCudaErrors(cudaMallocManaged((void**)&geom_manager, sizeof(geometry_manager)));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    create_geometries(geom_manager, mat_manager->get_device_materials());

    std::cout << "after create geoms" << std::endl;

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    rtweekend::clock c;

    c.start();
    render<<<blocks, threads>>>(cam, framebuffer, geom_manager->get_world(), rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    c.end();
    c.print();

    std::cout << "Saving framebuffer to file..." << std::endl;
    save_framebuffer_to_file(framebuffer, cam.res_x, res_y, "output/output.ppm");

    // freeing stuff
    checkCudaErrors(cudaDeviceSynchronize());
    // free_world<<<1,1>>>(d_list,d_world);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaFree(d_world));
    // checkCudaErrors(cudaFree(d_list));
    // checkCudaErrors(cudaFree(framebuffer));
    return 0;
}


