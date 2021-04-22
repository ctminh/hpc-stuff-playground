#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sys/syscall.h>
#include <atomic>
#include <chrono>

// for mpi, omp
#include <omp.h>
#include <mpi.h>

// for thallium
#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <bits/stdc++.h>

// for boost-serialize data
#include <boost/serialization/string.hpp>

// declare some namespaces
namespace tl = thallium;


// ================================================================================
// Global Variables
// ================================================================================
const int DEFAULT_NUM_TASKS = 10;

// ================================================================================
// Struct Definition
// ================================================================================
struct mat_task {
    int size;
    double *A;  // ptr to the allocated matrix A
    double *B;  // ptr to the allocated matrix B
    double *C;  // ptr to the allocated matrix C - result
};

// ================================================================================
// Util-functions
// ================================================================================
void initialize_matrix_rando(double *mat_ptr, int size){
    double low_bnd = 0.0;
    double upp_bnd = 10.0;
    std::uniform_real_distribution<double> ur_dist(low_bnd, upp_bnd);
    std::default_random_engine dre;
    for (int i = 0; i < size*size; i++){
        mat_ptr[i] = ur_dist(dre);
    }
}

void initialize_matrix_zeros(double *mat_ptr, int size){
    for (int i = 0; i < size*size; i++){
        mat_ptr[i] = 0.0;
    }
}

void compute_mxm(double *a, double *b, double *c, int size){
    
    // main loop to compute as a serial way
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i*size + j] = 0;
            for (int k = 0; k < size; k++) {
                c[i*size + j] += a[i*size + k] * b[k*size + j];
            }
        }
    }
}

void mxm_kernel(double *A, double *B, double *C, int size){

    // call the compute entry
    compute_mxm(A, B, C, size);
}

// ================================================================================
// Main function
// ================================================================================
int main(int argc, char **argv){

    // variables for tracking mpi-processes
    int my_rank;
    int num_ranks;
    int provided;   // level of provided thread support
    int requested = MPI_THREAD_MULTIPLE;    // level of desired thread support

    // init MPI at runtime
    MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    /*
     * **************************************************************************** 
     * Init thallium rpc-rdma data transfer
     * ****************************************************************************
     */
    
    // if rank = 0, init thallium server
    if (my_rank == 0){
        // check the server
        std::cout << "R0 is initializing the tl-server..." << std::endl;

        // init the tl-server mode
        tl::engine ser_engine("tcp", THALLIUM_SERVER_MODE);
        std::cout << "Init tl-server on R0 at " << ser_engine.self() << std::endl;
        std::string str_serveraddr = ser_engine.self();
        std::cout << "Cast the addr to string-type: " << str_serveraddr << std::endl;

        // use mpi_send to let the client know the server address


    } else if (my_rank == 1) {
        // check the client
        std::cout << "R1 is initializing the tl-client..." << std::endl;

        // get the ser-addr over mpi-transfer
        std::string ser_addr;

        // init the tl-client mode
        tl::engine cli_engine("tcp", MARGO_CLIENT_MODE);
        tl::remote_procedure remote_do_rdma = cli_engine.define("do_rdma").disable_response();
        tl::endpoint ser_endpoint = cli_engine.lookup(ser_addr);
    }
    

    /*
     * ****************************************************************************
     */

    // declare range of matrix sizes
    // int mat_size_arr[7] = {128, 256, 512, 640, 768, 896, 1024};
    int mat_size_arr[12] = {64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416};
    std::vector<int> total_load_per_rank_arr;
    total_load_per_rank_arr.resize(num_ranks);

    // get num required tasks per rank
    int num_tasks;
    if (argc > 2) {
        if (my_rank == 0)
            std::cout << "Note: using the number of created tasks by user..." << std::endl;
        num_tasks = atoi(argv[my_rank + 1]);
        std::cout << "Info: R" << my_rank << " creats " << num_tasks << " tasks!" << std::endl;
    } else {
        if (my_rank == 0)
            std::cout << "Note: creating " << DEFAULT_NUM_TASKS << " tasks per rank as default..." << std::endl;
        num_tasks = DEFAULT_NUM_TASKS;
    }

    // create different-sizes for each task
    const int n = num_tasks;
    const int num_sizes = 12;
    int list_mat_sizes[n];
    std::random_device rd;
    std::uniform_int_distribution<int> matsize_distribution(0, num_sizes-1);
    for (int i = 0; i < n; i++){
        int s_idx = matsize_distribution(rd);
        list_mat_sizes[i] = mat_size_arr[s_idx];
    }

    // check the list of matsize indices
    // std::cout << "R" << my_rank << ": ";
    // for (int j = 0; j < n; j++){
    //     std::cout << list_mat_sizes[j] << " ";
    // } std::cout << std::endl;

    // create a vector of matrices/tasks per rank
    // another reason to track them and delete later
    std::vector<mat_task> matrix_task_list;

    // create and init data for matrices
    // #pragma omp parallel for
    for (int i = 0; i < num_tasks; i++) {
        // create a mat_task instance
        int i_size = list_mat_sizes[i];
        struct mat_task t;
        t.size = i_size;
        t.A = new double[i_size * i_size];
        t.B = new double[i_size * i_size];
        t.C = new double[i_size * i_size];

        // init the matrix
        initialize_matrix_rando(t.A, i_size);
        initialize_matrix_rando(t.A, i_size);
        initialize_matrix_zeros(t.C, i_size);

        // add task to the matrix_arr
        matrix_task_list.push_back(t);
    }

    // set a barrier to make safe
    MPI_Barrier(MPI_COMM_WORLD);

    // create tasks by chameleon
    for (int i = 0; i < num_tasks; i++){
        // indicate the inputs of each task
        int m_size = matrix_task_list[i].size;
        double *A = matrix_task_list[i].A;
        double *B = matrix_task_list[i].B;
        double *C = matrix_task_list[i].C;

        // execute tasks
        mxm_kernel(A, B, C, m_size);

    }
    MPI_Barrier(MPI_COMM_WORLD);

    // show stats information afterall
    std::cout << "R" << my_rank << ": total_load = " << total_load_per_rank_arr[my_rank] << "(by mat_sizes!" << std::endl;
    std::cout << "R" << my_rank << ": total_load = " << 0.0 << "(by exe_time!" << std::endl;

    // free memory allocated by new
    for (int i = 0; i < num_tasks; i++){
        delete matrix_task_list[i].A;
        delete matrix_task_list[i].B;
        delete matrix_task_list[i].C;
    }

    // finalize mpi
    MPI_Finalize();

    return 0;
}