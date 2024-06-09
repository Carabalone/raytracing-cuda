#ifndef OPTIONAL_H
#define OPTIONAL_H

template <typename T>
class Optional {
private:
    T* value;
    bool has_value;

public:
    __device__ Optional() : has_value(false), value(nullptr) {}
    __device__ Optional(T _val) : value(&_val), has_value(true) {}

     __device__ bool hasValue() const { return has_value; }

    __device__ T& get() { 
        return *value; 
    }

    __device__ const T& get() const { 
        return *value; 
    }

    __device__ void reset() { 
        value = nullptr; 
        has_value = false; 
    }

    __device__ void set(T& new_value) {
        value = &new_value;
        has_value = true;
    }

};

#endif // OPTIONAL_H
