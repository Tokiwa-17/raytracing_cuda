#ifndef SPHERE_H
#define SPHERE_H
#include "hitable.h"
#include "vec3.h"
#include "ray.h"

class sphere : public hitable {
    public:
        __device__  sphere() {}
        __device__ sphere(vec3 center, float r) : center(center), radius(r) {}
        __device__ bool hit(const ray &r, float t_min, float t_max, hit_record& rec) const;
    public:
        vec3 center;
        float radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }
    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal =  (rec.p - center) / radius;
    return true;
}

#endif