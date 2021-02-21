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
    audo task_clean_up = taskflow.emplace([&](){
        for (int k = 0; k < K; ++k){
            sx[k] = 0.0f;
            sy[k] = 0.0f;
            c[k] = 0;
        }
    }).name("task_clean_up");

    // the main task for updating centroids
    td::Task pf;
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
            c[best_ks] += 1;
        }

        for (int k = 0; k < K; ++k){
            auto count = max(1, c[k]);
            mx[k] = sx[k] / count;
            my[k] = sy[k] / count;
        }
    }).name("task_update_centroids");

    // conditions for executing tasks
    auto condition = taskflow.emplace([m=0, M]() mutable {
        return (m++ < M) ? 0 : 1
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

    return {mx, my};
}   

// ----------------------------------------------------------------------------
// GPU implementation
// ----------------------------------------------------------------------------


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

    return 0;
}