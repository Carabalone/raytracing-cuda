#ifndef VEC3_H
#define VEC3_H

#include <iostream>

class vec3 {

public:
    float coords[3];

    __host__ __device__ vec3() : coords{0.0f, 0.0f, 0.0f} {};
    __host__ __device__ vec3(float e0, float e1, float e2) : coords{e0, e1, e2} {};

    __host__ __device__ float x() const { return coords[0]; }
    __host__ __device__ float y() const { return coords[1]; }
    __host__ __device__ float z() const { return coords[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-coords[0], -coords[1], -coords[2]); }
    __host__ __device__ float operator[](int i) const { return coords[i]; }
    __host__ __device__ float& operator[](int i) { return coords[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        coords[0] += v.coords[0];
        coords[1] += v.coords[1];
        coords[2] += v.coords[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(float t) {
        coords[0] *= t;
        coords[1] *= t;
        coords[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(float t) {
        return *this *= 1/t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2];
    }
};

using point3 = vec3;

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] + v.coords[0], u.coords[1] + v.coords[1], u.coords[2] + v.coords[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] - v.coords[0], u.coords[1] - v.coords[1], u.coords[2] - v.coords[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.coords[0] * v.coords[0], u.coords[1] * v.coords[1], u.coords[2] * v.coords[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, float f) {
    return vec3(u.coords[0] * f, u.coords[1] * f, u.coords[2] * f);
}

__host__ __device__ inline vec3 operator*(float f, const vec3& u) {
    return u * f;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.coords[0] * v.coords[0]
         + u.coords[1] * v.coords[1]
         + u.coords[2] * v.coords[2];
}

__host__ __device__ inline bool operator==(const vec3& u, const vec3& v) {
    return (
        u.coords[0] == v.coords[0] &&
        u.coords[1] == v.coords[1] &&
        u.coords[2] == v.coords[2]
    );
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.coords[1] * v.coords[2] - u.coords[2] * v.coords[1],
                u.coords[2] * v.coords[0] - u.coords[0] * v.coords[2],
                u.coords[0] * v.coords[1] - u.coords[1] * v.coords[0]);
}

__host__ __device__ inline vec3 normalize(const vec3& v) {
    return v / v.length();
}

#endif // VEC3_H
