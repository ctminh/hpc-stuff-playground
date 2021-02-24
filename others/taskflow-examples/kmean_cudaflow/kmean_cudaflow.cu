// This program implements the k-means clustering algorithm in three forms:
//  - sequential cpu
//  - parallel cpu
//  - gpu with conditional tasking
//  - gpu without conditional tasking

#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>

#include <iomanip>
#include <cfloat>
#include <climits>

// define L2 computation
#define L2(x1, y1, x2, y2) ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

// ----------------------------------------------------------------------------
// CPU (sequential) implementation
// ----------------------------------------------------------------------------
// run k-means on cpu
std::pair<std::vector<float>, std::vector<float>> cpu_seq(
    const int N, const int K, const int M,
    const std::vector<float>& px,
    const std::vector<float>& py)
{
    // clare some vectors
    std::vector<int> c(K);  // vector c: K elements (K groups)
    std::vector<float> sx(K), sy(K), mx(K), my(K);

    // initial centroids for each cluster/group
    for (int i = 0; i < K; i++){
        mx[i] = px[i];
        my[i] = py[i];
    }

    // loop for all iterations
    for (int iter = 0; iter < M; iter++){

        // clear the statistics
        for (int k = 0; k < K; k++){
            sx[k] = 0.0f;
            sy[k] = 0.0f;
            c[k]  = 0;
        }

        // find the best cluster-id for each points
        // loop: check all points, calculate the distance
        for (int i = 0; i < N; i++){
            float x = px[i];
            float y = py[i];
            float best_distance = std::numeric_limits<float>::max();    // just to assign a big value
            int best_k = 0;

            for (int k = 0; k < K; k++){
                const float d = L2(x, y, mx[k], my[k]);
                if (d < best_distance){
                    best_distance = d;
                    best_k = k;
                }
            }
        
            // gather all points belong to a cluster
            sx[best_k] += x;
            sy[best_k] += y;
            c[best_k] += 1;
        }

        // update the centroids
        for (int k = 0; k < K; k++){
            const int count = max(1, c[k]);
            mx[k] = sx[k] / count;
            my[k] = sy[k] / count;
        }
    }

    return {mx, my};
}

// ----------------------------------------------------------------------------
// CPU (parallel) implementation
// ----------------------------------------------------------------------------
std::pair<std::vector<float>, std::vector<float>> cpu_par(
    const int N, const int K, const int M,
    const std::vector<float>& px,
    const std::vector<float>& py)
{
    const auto num_threads = std::thread::hardware_concurrency();

    tf::Executor executor;
    tf::Taskflow taskflow("kmean-cpu-parallel");

    std::vector<int> c(K), best_ks(N);
    std::vector<float> sx(K), sy(K), mx(K), my(K);

    // initial centroids for each cluster/group
    // define this as a task
    auto task_init = taskflow.emplace([&](){
        for (int i = 0; i < K; i++){
            mx[i] = px[i];
            my[i] = py[i];
        }
    }).name("task_init");

    // clean the working vectors for storing values
    auto task_clean_up = taskflow.emplace([&](){
        for (int k = 0; k < K; ++k){
            sx[k] = 0.0f;
            sy[k] = 0.0f;
            c[k] = 0;
        }
    }).name("task_clean_up");

    // the main task for updating centroids
    tf::Task pf;
    // define this as a for-par-loop task (like omp_for)
    pf = taskflow.for_each_index(0, N, 1, [&](int i){
        float x = px[i];
        float y = py[i];
        float best_distance = std::numeric_limits<float>::max();
        int best_k = 0;
        for (int k = 0; k < K; ++k){
            const float d = L2(x, y, mx[k], my[k]);
            if (d < best_distance){
                best_distance = d;
                best_k = k;
            }
        }

        // just store indices to mark which points belong to which cluster
        best_ks[i] = best_k;
    });

    // name this task as par-for-loop-task
    pf.name("task_par_for_loop");

    // this task stores the coordinates of points belong to a cluster
    auto task_update_cluster = taskflow.emplace([&](){
        // traverse all points again
        for (int i = 0; i < N; i++){
            sx[best_ks[i]] += px[i];    // best_ks[i] indicates the cluster of the point i
            sy[best_ks[i]] += py[i];
            c[best_ks[i]] += 1;
        }

        for (int k = 0; k < K; ++k){
            auto count = max(1, c[k]);
            mx[k] = sx[k] / count;
            my[k] = sy[k] / count;
        }
    }).name("task_update_centroids");

    // conditions for executing tasks
    auto condition = taskflow.emplace([m=0, M]() mutable {
        return (m++ < M) ? 0 : 1;
    }).name("task_check_converged");

    // describe the order of executing tasks
    /* order for executions
        task_init() -->
        condition {
            task_clean_up() -->
            pf  -->
            task_update_cluster -->
        } */
    task_init.precede(task_clean_up);
    task_clean_up.precede(pf);
    pf.precede(task_update_cluster);
    condition.precede(task_clean_up)
            .succeed(task_update_cluster);
        
    // execute the taskflow
    executor.run(taskflow).wait();

    // dump the taskflow
    // taskflow.dump(std::cout);

    return {mx, my};
}   

// ----------------------------------------------------------------------------
// GPU implementation
// ----------------------------------------------------------------------------

/* Each point (thread) computes its distance to each centroid 
and adds its x and y values to the sum of its closest
centroid, as well as incrementing that centroid's count of assigned points. */
__global__ void assign_clusters(
    const float *px, const float *py,
    int N, const float *mx, const float *my,
    float *sx, float *sy, int k, int *c)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N){
        return;
    }

    // make global loads
    const float x = px[index];
    const float y = py[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int c = 0; c < k; ++c){
        const float distance = L2(x, y, mx[c], my[c]);
        if (distance < best_distance){
            best_distance = distance;
            best_cluster = c;
        }
    }

    // update clusters/assign clusters to compute the centroids
    atomicAdd(&sx[best_cluster], x);
    atomicAdd(&sy[best_cluster], y);
    atomicAdd(&c[best_cluster], 1);
}

/* Each thread is one cluster, which just recomputes its coordinates as the mean
 of all points assigned to it. */
__global__ void compute_new_means(
    float *mx, float *my,
    const float *sx, const float *sy, const int *c)
{
    const int cluster = threadIdx.x;
    const int count = max(1, c[cluster]);
    mx[cluster] = sx[cluster] / count;
    my[cluster] = sy[cluster] / count;
}


std::pair<std::vector<float>, std::vector<float>> gpu_cond_tasks(
    const int N, const int K, const int M,
    const std::vector<float> &h_px,
    const std::vector<float> &h_py)
{
    std::vector<float> h_mx, h_my;  // contains the returned centroids
    float *d_px, *d_py, *d_mx, *d_my, *d_sx, *d_sy, *d_c;   // mx, my for keeping centroids/cluster per iter

    // copy values of all points to host_mx, _my
    for (int i = 0; i < K; i++){
        h_mx.push_back(h_px[i]);
        h_my.push_back(h_py[i]);
    }

    // create a taskflow graph
    tf::Executor executor;
    tf::Taskflow taskflow("K-Means-GPU-without-cond-tasks");

    // tasks for allocating mem on GPU devices
    auto allocate_px = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_px, N*sizeof(float)), "failed to allocate d_px");
    }).name("allocate_px_gpu");
    auto allocate_py = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_py, N*sizeof(float)), "failed to allocate d_py");
    }).name("allocate_py_gpu");
    auto allocate_mx = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_mx, K*sizeof(float)), "failed to allocate d_mx");
    }).name("allocate_mx_gpu");
    auto allocate_my = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_my, K*sizeof(float)), "failed to allocate d_my");
    }).name("allocate_my_gpu");
    auto allocate_sx = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_sx, K*sizeof(float)), "failed to allocate d_sx"); 
    }).name("allocate_sx_gpu");
    auto allocate_sy = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_sy, K*sizeof(float)), "failed to allocate d_sy"); 
    }).name("allocate_sy_gpu");
    auto allocate_c = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaMalloc(&d_c, K*sizeof(float)), "failed to allocate d_c");
    }).name("allocate_c_gpu");

    // task for copying data from host to device
    auto h2d = taskflow.emplace([&](tf::cudaFlow &cf){
        cf.copy(d_px, h_px.data(), N).name("h2d_px"); // cp data from h_px to d_px
        cf.copy(d_py, h_py.data(), N).name("h2d_py");
        cf.copy(d_mx, h_mx.data(), N).name("h2d_dx");
        cf.copy(d_my, h_my.data(), N).name("h2d_dy");
    }).name("h2d_copy_data");

    // the computation task
    auto kmeans = taskflow.emplace([&](tf::cudaFlow &cf){
        // init data on GPU
        auto zero_c = cf.zero(d_c, K).name("zero_c");
        auto zero_sx = cf.zero(d_sx, K).name("zero_sx");
        auto zero_sy = cf.zero(d_sy, K).name("zero_sy");

        auto cluster = cf.kernel((N+1024-1)/1024, 1024, 0,
                                assign_clusters,
                                d_px, d_py, N, d_mx, d_my, d_sx, d_sy, K, d_c).name("assign_clus_kernel");

        auto new_centroids = cf.kernel(
            1, K, 0,
            compute_new_means,
            d_mx, d_my, d_sx, d_sy, d_c).name("update_centroids_kernel");
        
        // decide the order for executing the main tasks
        cluster.precede(new_centroids)
                .succeed(zero_c, zero_sx, zero_sy);
    }).name("update_means_gpu");


    // declare the conditions for tasks
    auto condition = taskflow.emplace([i=0, M]() mutable {
        return i++ < M ? 0 : 1;
    }).name("check_converged");

    // the task, copying data back from GPU to host
    auto stop = taskflow.emplace([&](tf::cudaFlow &cf){
        cf.copy(h_mx.data(), d_mx, K).name("d2h_mx");
        cf.copy(h_my.data(), d_my, K).name("d2h_my");
    }).name("d2h_copy_back_data");

    // free memory task
    auto free = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(d_px), "failed to free d_px");
        TF_CHECK_CUDA(cudaFree(d_py), "failed to free d_py");
        TF_CHECK_CUDA(cudaFree(d_mx), "failed to free d_mx");
        TF_CHECK_CUDA(cudaFree(d_my), "failed to free d_my");
        TF_CHECK_CUDA(cudaFree(d_sx), "failed to free d_sx");
        TF_CHECK_CUDA(cudaFree(d_sy), "failed to free d_sy");
        TF_CHECK_CUDA(cudaFree(d_c),  "failed to free d_c");
    }).name("free_mem_on_gpu");

    // make the global orders
    h2d.succeed(allocate_px, allocate_py, allocate_mx, allocate_my);
    kmeans.succeed(allocate_sx, allocate_sy, allocate_c, h2d)
            .precede(condition);
    condition.precede(kmeans, stop);
    stop.precede(free);

    // run the taskflow
    executor.run(taskflow).wait();

    //std::cout << "dumping kmeans graph ...\n";
    taskflow.dump(std::cout);
    return {h_mx, h_my};
}


// ----------------------------------------------------------------------------
// MAIN function
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
    // check num of arguments
    if(argc != 4) {
        std::cerr << "usage: ./kmean_cudaflow num_points k num_iterations\n";
        std::exit(EXIT_FAILURE);
    }

    // get args
    const int N = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    const int M = std::atoi(argv[3]);

    // conditions for each arguments
    if(N < 1) {
        throw std::runtime_error("num_points must be at least one");
    }
    
    if(K >= N) {
        throw std::runtime_error("k must be smaller than the number of points");
    }
    
    if(M < 1) {
        throw std::runtime_error("num_iterations must be larger than 0");
    }


    // declare arrays
    std::vector<float> h_px, h_py, mx, my;

    // Randomly generate N points
    std::cout << "generating " << N << " random points ...\n";
    for(int i=0; i<N; ++i) {
        h_px.push_back(rand()%1000 - 500);
        h_py.push_back(rand()%1000 - 500);
    }

    // ----------------- k-means on cpu_seq
    std::cout << "running k-means on cpu (sequential) ... ";
    // start_time
    auto sbeg = std::chrono::steady_clock::now();
    // call cpu_kmean_kernel: std::tie is to create a tuple of values
    std::tie(mx, my) = cpu_seq(N, K, M, h_px, h_py);
    // end_time
    auto send = std::chrono::steady_clock::now();
    // show results
    std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(send-sbeg).count()
            << " ms\n";
    std::cout << "k centroids found by cpu (sequential)\n";
    for(int k = 0; k < K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';
    }


    // ---------------- k-mean cpu parallel
    std::cout << "running k-means on cpu (parallel) ... ";
    auto p_beg_time = std::chrono::steady_clock::now();
    std::tie(mx, my) = cpu_par(N, K, M, h_px, h_py);
    auto p_end_time = std::chrono::steady_clock::now();
    std::cout << "completed with " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(p_end_time-p_beg_time).count()
            << " ms\n";
  
    std::cout << "k centroids found by cpu (parallel)\n";
    for(int k=0; k<K; ++k) {
    std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                          << std::setw(10) << my[k] << '\n';  
    }


    // ----------------- k-means on gpu with conditional tasking
    std::cout << "running k-means on GPU (with conditional tasking) ...";
    auto pgpu_con_beg_time = std::chrono::steady_clock::now();
    std::tie(mx, my) = gpu_cond_tasks(N, K, M, h_px, h_py);
    auto pgpu_con_end_time = std::chrono::steady_clock::now();
    std::cout << "completed with "
            << std::chrono::duration_cast<std::chrono::milliseconds>(pgpu_con_end_time-pgpu_con_beg_time).count()
            << " ms\n";

    return 0;
}