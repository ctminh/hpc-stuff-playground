#include "multiple-apps.hpp"

#ifndef MAT_MUL_APP
#define MAT_MUL_APP 0
#endif

#ifndef JACOBI_APP
#define JACOBI_APP 0
#endif

#ifndef CHOLESKY_APP
#define CHOLESKY_APP 1
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

    numberOfTasks = atoi(argv[1]);
	
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
	// init matrix A
    double **chol_A;
    chol_A = new double*[numberOfTasks];
    int chol_mat_size = 4;
	int tile_size = 2;
	const int nt = chol_mat_size/tile_size;

    #pragma omp parallel for
    for (int i = 0; i < numberOfTasks; i++){
        chol_A[i] = new double[chol_mat_size*chol_mat_size];
        // init data for chol_mat
        init_chol_mat(chol_A[i], chol_mat_size);
    }
	
	// check the matrix of the first task
	printf("The matrix of the first task:\n");
	print_chol_mat(chol_A[0], chol_mat_size);
	printf("-------------------------------\n");
		
	// create origin_mat_A
	double **chol_orig_A;
	chol_orig_A = new double*[numberOfTasks];
	
	// create blocked_matrix
	double **chol_blocked_A;
	chol_blocked_A = new double*[numberOfTasks];
	for (int i = 0; i < numberOfTasks; i++){
		chol_blocked_A[i] = new double[nt*nt];
	}
	
	// store origin matrix
	for (int i = 0; i < numberOfTasks; i++){
		for (int j = 0; j < chol_mat_size*chol_mat_size; j++){
			chol_orig_A[i][j] = chol_A[i][j];
		}
	}
	
	// check the origin_matrix of the first task
	printf("The origin_matrix of the first task:\n");
	print_chol_mat(chol_orig_A[0], chol_mat_size);
	printf("-------------------------------\n");
	
	// convert to blocks
	for (int i = 0; i < numberOfTasks; i++){
		convert_to_blocks(tile_size, nt, chol_mat_size, chol_A[i], chol_blocked_A[i]);
	}
	
#endif

    // wait for other processes
    MPI_Barrier(MPI_COMM_WORLD);

    // set time-measurement
    fTimeStart = MPI_Wtime();

    printf("R%d, run mat_mul tasks\n", iMyRank);
//    create_mat_mul_tasks(numberOfTasks, matrices_a, matrices_b, matrices_c, matrixSize);

//     if (iMyRank < (iNumProcs/2))
//     {
// #if MAT_MUL_APP
//         printf("R%d, run mat_mul tasks\n", iMyRank);
//         create_mat_mul_tasks(numberOfTasks, matrices_a, matrices_b, matrices_c, matrixSize);
// #endif
//     } else {
// #if JACOBI_APP
//         printf("R%d, run jacobi tasks\n", iMyRank);
//         create_jacobi_tasks(numberOfTasks, jaco_A, jaco_b, jaco_x, jaco_xtmp, jaco_mat_size);
// #endif
//     }

    // distribute and execute all tasks
//    int res = chameleon_distributed_taskwait(0);
    // #pragma omp single
    // MPI_Barrier(MPI_COMM_WORLD);


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

    MPI_Finalize();
    return 0;
}
