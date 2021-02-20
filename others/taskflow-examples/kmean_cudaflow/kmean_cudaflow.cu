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

        // clear the points of each group
        for (int k = 0; k < K; k++){
            sx[k] = 0.0f;
            sy[k] = 0.0f;
            c[k]  = 0;
        }

        // find the best cluster-id for each points
        // loop: check all points, calculate the distance
        
    }
    
}

// ----------------------------------------------------------------------------
// CPU (parallel) implementation
// ----------------------------------------------------------------------------


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

    // k-means on cpu_seq
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
    for(int k=0; k<K; ++k) {
        std::cout << "centroid " << k << ": " << std::setw(10) << mx[k] << ' ' 
                                            << std::setw(10) << my[k] << '\n';  
    }

    return 0;
}