#ifndef GEOMETRY_MANAGER_H
#define GEOMETRY_MANAGER_H

#include <vector>
#include "utility.cuh"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"

enum geometry_type {
    sphere_t, list_t
};

struct geometry_info {
    geometry_type type;
    point3 center;
    float radius;
    material* mat;
    int size;
    hittable** list;
};

class geometry_manager {
private:
    std::vector<geometry_info*> geometries;
    geometry_info** d_geometry_info;
    hittable** d_geometries;
    hittable_list* d_hlist;
    hittable** d_world; 
    int count = -1;

public:
    geometry_manager() : geometries(std::vector<geometry_info*>()), d_geometries(nullptr), d_geometry_info(nullptr) {}

    __host__ __device__ void print_geom_info(geometry_info* geom) {


        printf("\n-----------------------------------------------\n");
        switch(geom->type) {
            case sphere_t:
                printf("Geometry: Sphere\n");
                printf("Center: (%f, %f, %f)\n", geom->center.x(), geom->center.y(), geom->center.z());
                printf("Radius: %f\n", geom->radius);
                printf("Material: %p\n", geom->mat);
                break;
            case list_t:
                printf("Geometry: Hittable List\n");
                printf("Number of hittables: %d\n", geom->size);
                break;
        }
    }

    __host__ void add_geometry(geometry_info* geom) {
        geometries.push_back(geom);
    }

    __host__ void dispatch() {
        int num_geometries = geometries.size();
        printf("num_geometries: %d\n", num_geometries);
        count = num_geometries;

        checkCudaErrors(cudaMalloc((void**) &d_geometry_info, num_geometries * sizeof(geometry_info*)));
        checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

        for (int i = 0; i < num_geometries; i++) {
            geometry_info* d_geom_info;
            size_t size = sizeof(geometry_info);
            checkCudaErrors(cudaMalloc((void**) &d_geom_info, size));
            checkCudaErrors(cudaMemcpy(d_geom_info, geometries[i], size, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&d_geometry_info[i], &d_geom_info, sizeof(geometry_info*), cudaMemcpyHostToDevice));
        }

        checkCudaErrors(cudaMalloc((void**)&d_geometries, count * sizeof(hittable*)));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        for (auto& geom : geometries) {
            delete geom;
        }

        geometries.clear();
    }

    __host__ __device__ hittable** get_device_geometries() const {
        return d_geometries;
    }

    geometry_info** get_device_geometry_info() const {
        return d_geometry_info;
    }

    __host__ __device__ int size() const {
        return count;
    }

    __host__ __device__ hittable_list* get_hlist() {
        return d_hlist;
    }


    __host__ __device__ hittable** get_world() {
        return d_world;
    }
};

__global__ void create_device_geometries(geometry_info** d_geometry_info, hittable** d_geometries, hittable_list* h_list, hittable** d_world, int count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello I'm executing\n");
        for (int i = 0; i < count; i++) {
            geometry_info* geom_info = d_geometry_info[i];

            switch (geom_info->type) {
                case sphere_t:
                    printf("Creating sphere\n");
                    d_geometries[i] = new sphere(geom_info->center, geom_info->radius, geom_info->mat);
                    break;
                case list_t:
                    printf("Creating list\n");
                    d_geometries[i] = new hittable_list(geom_info->list, geom_info->size);
                    break;
            }
        }

        h_list = new hittable_list(d_geometries, count);
        *d_world = h_list;
    }
}

#endif // GEOMETRY_MANAGER_H
