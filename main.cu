#include <iostream>
#include <chrono>
#include "vec3.h" 
#include "ray.h"

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

__device__ color ray_color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = (unit_direction.y() + 1.0f) * 0.5;
    return vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
}

__global__ void render(vec3 *fb, int max_x, int max_y) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;
    int idx = y * max_x + x;
    fb[idx] = vec3(float(x) / (max_x - 1), \
                   float(y) / (max_y - 1), \
                   0.25f);
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