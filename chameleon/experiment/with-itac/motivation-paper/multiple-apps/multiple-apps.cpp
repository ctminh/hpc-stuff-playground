#include "multiple-apps.hpp"

#ifndef MAT_MUL_APP
#define MAT_MUL_APP 1
#endif

#ifndef JACOBI_APP
#define JACOBI_APP 1
#endif

#ifndef CHOLESKY_APP
#define CHOLESKY_APP 0
#endif

// some definition
enum app {
    mat_mul = 0,
    jacobi = 1,
    cholesky = 2
};

/* ////////////////////// Main-Functions /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////// */

int main(int argc, char **argv)
{
	int iMyRank, iNumProcs;
	int provided;
	int requested = MPI_THREAD_MULTIPLE;
	MPI_Init_thread(&argc, &argv, requested, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &iNumProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
	int numberOfTasks;
	double fTimeStart, fTimeEnd;
	double wTimeCham, wTimeHost;
	bool pass = true;

    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);

    // check arguments
    if(argc > 2) {
        if(iMyRank == 0) {
            LOG(iMyRank, "using user-defined initial load distribution...");    
        } 
        numberOfTasks = atoi(argv[iMyRank + 1]);
    } else { 
        printHelpMessage();
        return 0;     
    }

    /*///////////////////// init matrix-multiplication task /////////////////////*/
    // range of matrix sizes
    // int range_size[5] = {128, 256, 512, 1024, 2048};
    // int total_load[2] = {0, 0};
    // srand(time(NULL));  // seed to random generation

    // // create different size tasks array
    // int **mat_size_idx_arr;
    // mat_size_idx_arr = new int*[iNumProcs];
    // for (int i = 0; i < iNumProcs; i++){
    //     mat_size_idx_arr[i] = new int[numberOfTasks];
    //     for (int j = 0; j < numberOfTasks; j++)
    //         mat_size_idx_arr[i][j] = 0;
    // }
    // create_diff_task_sizes(iMyRank, numberOfTasks, iNumProcs, mat_size_idx_arr);
    // MPI_Barrier(MPI_COMM_WORLD);

    // if (iMyRank == 1){
    //     for (int i = 0; i < numberOfTasks; i++){
    //         mat_size_idx_arr[iMyRank][i] = mat_size_idx_arr[0][numberOfTasks-i-1];  // result of this phase is mat_size_idx_arr[Rank][task-ith] = size
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
	
    // log the info
    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());

#if MAT_MUL_APP
    // 1. init input-data for mat-mul process -----------------------------
	double **matrices_a, **matrices_b, **matrices_c;
	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];
    int matrixSize[numberOfTasks];

    #pragma omp parallel for
	for (int i = 0; i < numberOfTasks; i++) {
        // create random matrix_sizes, matrixSize is set above, just assign again
        // matrixSize[i] = range_size[mat_size_idx_arr[iMyRank][i]];
        matrixSize[i] = 1024;    // just for testing the tool
        // total_load[iMyRank] += matrixSize[i];

 		matrices_a[i] = new double[(long)matrixSize[i]*matrixSize[i]];
    	matrices_b[i] = new double[(long)matrixSize[i]*matrixSize[i]];
    	matrices_c[i] = new double[(long)matrixSize[i]*matrixSize[i]];

        initialize_matrix_test_A(matrices_a[i], matrixSize[i]);
        initialize_matrix_test_A(matrices_b[i], matrixSize[i]);
        initialize_matrix_zero(matrices_c[i], matrixSize[i]);
    }
#endif
    /////////////////////////////////////////////////////////////////////////

#if JACOBI_APP
    // 2. init input-data for jacobi --------------------------------------
    double **jaco_A, **jaco_b, **jaco_x, **jaco_xtmp;
    jaco_A    = new double*[numberOfTasks];
    jaco_b    = new double*[numberOfTasks];
    jaco_x    = new double*[numberOfTasks];
    jaco_xtmp = new double*[numberOfTasks];
    int jaco_mat_size = 512;

    #pragma omp parallel for
	for (int i = 0; i < numberOfTasks; i++) {
        jaco_A[i] = new double[jaco_mat_size*jaco_mat_size];
    	jaco_b[i] = new double[jaco_mat_size];
    	jaco_x[i] = new double[jaco_mat_size];
        jaco_xtmp[i] = new double[jaco_mat_size];
        // init data for jacobi mat
        init_jacobi_data(jaco_A[i], jaco_b[i], jaco_x[i], jaco_mat_size);
    }
#endif

#if CHOLESKY_APP
    double **chol_A;
    chol_A = new double*[numberOfTasks];
    int chol_mat_size = 4;

    #pragma omp parallel for
    for (int i = 0; i < numberOfTasks; i++){
        chol_A[i] = new double[chol_mat_size*chol_mat_size];
        // init data for chol_mat
    }
#endif

    // wait for other processes
    MPI_Barrier(MPI_COMM_WORLD);

    // set time-measurement
    fTimeStart = MPI_Wtime();

    if (iMyRank < (iNumProcs/2))
    {
#if MAT_MUL_APP
        printf("R%d, run mat_mul tasks\n", iMyRank);
        create_mat_mul_tasks(numberOfTasks, matrices_a, matrices_b, matrices_c, matrixSize);
#endif
    } else {
#if JACOBI_APP
        printf("R%d, run jacobi tasks\n", iMyRank);
        create_jacobi_tasks(numberOfTasks, jaco_A, jaco_b, jaco_x, jaco_xtmp, jaco_mat_size);
#endif
    }

    // distribute and execute all tasks
    int res = chameleon_distributed_taskwait(0);

    // get total load on each process
    // printf("Total load on R%d: %d\n", iMyRank, total_load[iMyRank]);

    // get execution_time for each process
    fTimeEnd = MPI_Wtime();
    wTimeCham = fTimeEnd - fTimeStart;

    // print process's execution time
    printf("#R%d: Computations with chameleon took %.5f\n", iMyRank, wTimeCham);

    // wait for other processes
    MPI_Barrier(MPI_COMM_WORLD);

#if MAT_MUL_APP
    // free memory for mat-mul
    for(int i = 0; i < numberOfTasks; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }
    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;
    // delete[] mat_size_idx_arr;
#endif

#if JACOBI_APP
    // free memory for jacobi
    for(int i = 0; i < numberOfTasks; i++) {
    	delete[] jaco_A[i];
    	delete[] jaco_b[i];
    	delete[] jaco_x[i];
        delete[] jaco_xtmp[i];
    }
    delete[] jaco_A;
    delete[] jaco_b;
    delete[] jaco_x; 
    delete[] jaco_xtmp;
#endif

    // wait for other processes
    MPI_Barrier(MPI_COMM_WORLD);

    #pragma omp parallel
    {
        chameleon_thread_finalize();
    }
    chameleon_finalize();

    MPI_Finalize();
    return 0;
}
