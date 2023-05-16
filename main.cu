#include <iostream>
#include <chrono>
#include "vec3.h" 
#include "ray.h"
#include "hitable.h"
#include "sphere.h"

const int nx = 1200, ny = 800;
const int tx = 16, ty = 16;
vec3 *fb;
constexpr const int num_pixels = nx * ny;

#define checkCudaErrors(status)                   \
    do                                            \
    {                                             \
        if (status != 0)                          \
        {                                         \
            fprintf(stderr, "CUDA failure at [%s] (%s:%d): %s\n", __PRETTY_FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(status)); \
            cudaDeviceReset();                    \
            abort();                              \
        }                                         \
    } while (0)


void prep() {
    checkCudaErrors(cudaMallocManaged(&fb, num_pixels * sizeof(vec3)));
}

__device__ float hit_sphere(const vec3 &center, float radius, const ray &r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return -1.0f;
    else return (-half_b - sqrt(discriminant)) / a;
}

__device__ color ray_color(const ray& r) {
    sphere sph{vec3(0.0f, 0.0f, -1.0f), 0.5};
    hit_record rec;
    float t = -1.0f;
    if (sph.hit(r, 0.0f, 1e5, rec)) t = rec.t;
    if (t > 0.0) {
        vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
        return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    float k = (unit_direction.y() + 1.0f) * 0.5;
    return vec3(1.0f, 1.0f, 1.0f) * (1.0f - k) + vec3(0.5f, 0.7f, 1.0f) * k;
}

__global__ void render(vec3 *fb, int max_x, int max_y) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;
    constexpr const float aspect_ratio = 16.0 / 9.0;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * aspect_ratio;
    float focal_length = 1.0f; 

    vec3 origin {0, 0, 0};
    vec3 horizontal {viewport_width, 0, 0};
    vec3 vertical {0, viewport_height, 0};
    vec3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float u = float(i) / max_x, v = float(j) / max_y;
    ray r {origin, lower_left_corner + horizontal * u + vertical * v - origin};
    color pixel_color = ray_color(r);
    int idx = y * max_x + x;
    fb[idx] = pixel_color;
}

int main() {

    prep();
    dim3 thread(tx, ty, 1);
    dim3 block((nx + tx - 1) / tx, (ny + ty - 1) / ty, 1);
    namespace ch = std::chrono;
    auto beg = ch::high_resolution_clock::now();
    render<<<block, thread>>>(fb, nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = ch::high_resolution_clock::now();
    double dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; // ms
    //std::cerr << dur << "ms" << std::endl;

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            float r = fb[pixel_index].x();
            float g = fb[pixel_index].y();
            float b = fb[pixel_index].z();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}