#ifndef MATERIAL_MANAGER_H
#define MATERIAL_MANAGER_H

#include <vector>
#include "material.cuh"
#include "utility.cuh"

class material_manager {
private:
    std::vector<material_info*> materials;
    material_info** d_material_info;
    material** d_materials;
    int count = -1;

public:
    material_manager() : materials(std::vector<material_info*>()), d_materials(nullptr), d_material_info(nullptr) {}

    __host__ __device__ void print_mat_info(material_info* mat) {

        printf("\n-----------------------------------------------\n");
        switch(mat->type) {
            case (material_type::lambertian_t):
                printf("is lambertian\n");
                printf("albedo: r: %.2f, g: %.2f, b: %.2f\n", mat->albedo.x(), mat->albedo.y(), mat->albedo.z());
                break;
            case (material_type::metal_t):
                printf("is metal\n");
                printf("albedo: r: %.2f, g: %.2f, b: %.2f\n", mat->albedo.x(), mat->albedo.y(), mat->albedo.z());
                printf("fuzz: %.2f\n", mat->fuzz);
                break;
            case (material_type::dieletric_t):
                printf("is dieletric\n");
                printf("refraction_index: %.2f\n", mat->refraction_index);
                break;
        }
    }

    __host__ void add_material(material_info* mat) {
        materials.push_back(mat);
    }

    __host__ void dispatch() {
        int num_materials = materials.size();
        printf("num_materials: %d\n", num_materials);
        count = num_materials;

        checkCudaErrors(cudaMalloc((void**) &d_material_info, num_materials * sizeof(material_info*)));

        for (int i = 0; i < num_materials; i++) {
            material_info* d_mat_info; 
            size_t size = sizeof(material_info);
            checkCudaErrors(cudaMalloc((void**) &d_mat_info, size));
            checkCudaErrors(cudaMemcpy(d_mat_info, materials[i], size, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(&d_material_info[i], &d_mat_info, sizeof(material_info*), cudaMemcpyHostToDevice));
        }

        checkCudaErrors(cudaMalloc((void**)&d_materials, count * sizeof(material*)));

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        for (auto& mat : materials) {
            delete mat;
        }

        materials.clear();
    }

    __host__ __device__ material** get_device_materials() const {
        return d_materials;
    }

    material_info** get_device_material_info() const {
        return d_material_info;
    }

    __host__ __device__ int size() const {
        return count;
    }
};

__global__ void create_device_materials(material_info** d_material_info, material** d_materials, int count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i=0; i < count; i++) {
            material_info* mat_info = d_material_info[i];

            switch(mat_info->type) {
                case (material_type::lambertian_t):
                    d_materials[i] = new lambertian(mat_info->albedo);
                    break;
                case (material_type::metal_t):
                    d_materials[i] = new metal(mat_info->albedo, mat_info->fuzz);
                    break;
                case (material_type::dieletric_t):
                    d_materials[i] = new dieletric(mat_info->refraction_index);
                    break;
            }
        }
    }
}

#endif  // MATERIAL_MANAGER_H
