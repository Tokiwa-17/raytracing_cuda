#ifndef HITABLE_H
#define HITABLE_H

#include "vec3.h"
struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
};

class hitable {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record &rec) const = 0;
};

#endif