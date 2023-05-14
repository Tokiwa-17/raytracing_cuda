#include <iostream>
#include <chrono>

const int nx = 1200, ny = 800;
const int tx = 16, ty = 16;
float *fb;
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
    checkCudaErrors(cudaMallocManaged(&fb, num_pixels * 3 * sizeof(float)));
}

__global__ void render(float *fb, int max_x, int max_y) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= max_x || y >= max_y) return;
    int idx = y * max_x * 3 + x * 3;
    fb[idx] = float(x) / (max_x - 1);
    fb[idx + 1] = float(y) / (max_y - 1);
    fb[idx + 2] = 0.25;
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
            size_t pixel_index = j * 3 * nx + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
    return 0;
}