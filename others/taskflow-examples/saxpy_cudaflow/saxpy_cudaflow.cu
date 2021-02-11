#include <taskflow/cudaflow.hpp

// saxpy (single-precision A·X Plus Y) kernel
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

// main functions begins
int main(int argc, char *argv[]){
    
    tf::Taskflow taskflow;
    tf::Executor executor;

    const unsigned N = 1 << 20;     // size of the vector

    std::vector<float> hx(N, 1.0f);     // vector x at host
    std::vector<float> hy(N, 2.0f);     // vector y at host

    // allocate memory on the devices
    float *dx{nullptr};
    float *dy{nullptr};

    tf::Task allocate_x = taskflow.emplace(
        [&](){ cudaMalloc(&dx, N * sizeof(float)); }
    ).name("allocate_x");

    tf::Task allocate_y = taskflow.emplace(
        [&](){ cudaMalloc(&dy, N * sizeof(float)); }
    ).name("allocate_y");

    // create a task flow for the program
    tf::Task cudaflow = taskflow.emplace([&](tf::cudaflow &cf) {
        // create data transfer tasks
        tf::cudaTask h2d_x = cf.copy(dx, hx.data(), N).name("h2d_x");
        tf::cudaTask h2d_y = cf.copy(dy, hx.data(), N).name("h2d_y");
        tf::cudaTask d2h_x = cf.copy(hx.data(), dx, N).name("d2h_x");
        tf::cudaTask d2h_y = cf.copy(hy.data(), dy, N).name("d2h_y");

        // task to run the compute kernal - saxpy
        // config: <<<(N+255)/256, 256, 0>>> (N, 2.0, dx, dy)
        tf::cudaTask kernel = cf.kernel(
            // thread-block config | comp-kernel |  arguments for the kernel
            (N+255)/256, 256, 0,     saxpy,         N, 2.0f, dx, dy
        ).name("saxpy_kernel");

        // description about the relationship of the kernel
        kernel.succeed(h2d_x, h2d_y)
                .precede(d2h_x, d2h_y);
    }).name("saxpy");

    // overlap memory alloc???
    cudaflow.succeed(allocate_x, allocate_y);

    executor.run(taskflow).wait();

    // dump the taskflow
    taskflow.dump(std::cout);

    return 0;
}
