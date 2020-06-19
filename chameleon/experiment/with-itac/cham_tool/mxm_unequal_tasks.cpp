#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#include <assert.h>
#include <mpi.h>
#include "chameleon.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include "math.h"
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>
#include <atomic>
#include <ctime>

#define SPEC_RESTRICT __restrict__
//#define SPEC_RESTRICT restrict

/* init matrix with random values */
void initialize_matrix_rnd(double *mat, int matrixSize) {
	double lower_bound = 0;
	double upper_bound = 10000;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;

	for(int i = 0; i < matrixSize*matrixSize; i++) {
		mat[i] = unif(re);
	}
}

/* init matrix 0 */
void initialize_matrix_zero(double *mat, int matrixSize) {
	for(int i = 0; i < matrixSize*matrixSize; i++) {
		mat[i] = 0;
	}
}

/* init matrix with value = 1 */
void initialize_matrix_test_A(double *mat, int matrixSize) {
	for(int i = 0; i < matrixSize*matrixSize; i++) {
			mat[i] = 1;
    }
}

/* the main task in experiment */
void compute_matrix_matrix(double * SPEC_RESTRICT a, double * SPEC_RESTRICT b, double * SPEC_RESTRICT c, int matrixSize) {
    // make the tasks more computational expensive by repeating this operation several times to better see effects
    for  (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            c[i*matrixSize + j] = 0;
            for (int k = 0; k < matrixSize; k++) {
                c[i*matrixSize + j] += a[i*matrixSize + k] * b[k*matrixSize + j];
            }
        }
    }
}

/* function to check the result */
bool check_test_matrix(double *c, double val, int matrixSize) {
	int iMyRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &iMyRank);
	for(int i = 0; i < matrixSize; i++) {
		for(int j = 0; j < matrixSize; j++) {
			if(fabs(c[i*matrixSize + j] - val) > 1e-3) {
				printf("#R%d (OS_TID:%ld): Error in matrix entry (%d,%d) expected:%f but value is %f\n", iMyRank, syscall(SYS_gettid),i,j,val,c[i*matrixSize+j]);
				return false;
			}
		}
	}
	return true;
}

void printHelpMessage() {
    std::cout<<"Usage: mpiexec -n np ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] "<<std::endl;
    std::cout<<"    Arguments: "<<std::endl;
    std::cout<<"        matrixSize: number of elements of the matrixSize x matrixSize matrices"<<std::endl;
    std::cout<<"        nt_(i): number of tasks for process i "<<std::endl;
    std::cout<<"If the number of tasks is not specified for every process, the application will generate an initial task distribution"<<std::endl; 
}

void printArray(int rank, double * SPEC_RESTRICT array, char* arr_name, int n) {
    printf("#R%d (OS_TID:%ld): %s[0-%d] at (" DPxMOD "): ", rank, syscall(SYS_gettid), arr_name, n, DPxPTR(&array[0]));
    for(int i = 0; i < n; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void matrixMatrixKernel(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int matrixSize, int i) {
    compute_matrix_matrix(A, B, C, matrixSize);
}

int *shuffle(size_t n){
    int *arr = new int[n];
    for (int i = 0; i < n; i++){
        arr[i] = i;
    }

    if (n > 1) 
    {
        size_t i;
        srand(time(NULL));
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
    return arr;
}

int create_diff_task_sizes(int iMyRank, int numberOfTasks, int iNumProcs, int **result_array)
{
    float load_array[5] = {0.2, 0.15, 0.3, 0.15, 0.2};   // 20% matrix size 128, 15% matrix size 256, ...
    int N = numberOfTasks;
    int l1 = load_array[0] * N;
    int l2 = l1 + load_array[1] * N;
    int l3 = l2 + load_array[2] * N;
    int l4 = l3 + load_array[3] * N;
    printf("[0, l1, l2, l3, l4, N] = [0, %d, %d, %d, %d, %d]\n", l1, l2, l3, l4, N);
    int *tmp_array = new int[numberOfTasks];
    for (int i = 0; i < numberOfTasks; i++){
        if (i < l1) tmp_array[i] = 0; //mat_size = 128
        else if (i >= l1 && i < l2) tmp_array[i] = 1;
        else if (i >= l2 && i < l3) tmp_array[i] = 2;
        else if (i >= l3 && i < l4) tmp_array[i] = 3;
        else tmp_array[i] = 4;
    }
    
    // create a random array to mix values in tmp_array
    int *shuffle_array = new int[numberOfTasks];
    shuffle_array = shuffle(numberOfTasks);

    for (int i = 0; i < numberOfTasks; i++){
        result_array[0][i] = tmp_array[shuffle_array[i]];
        result_array[1][i] = tmp_array[shuffle_array[i]];
    }

    delete[] shuffle_array;
    delete[] tmp_array;

    return 0;
}

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
   
    // range of matrix sizes
    int range_size[5] = {128, 256, 512, 1024, 2048};
    int total_load[2] = {0, 0};
    srand(time(NULL));  // seed to random generation

    #pragma omp parallel
    {
        chameleon_thread_init();
    }
    // necessary to be aware of binary base addresses to calculate offset for target entry functions
    chameleon_determine_base_addresses((void *)&main);

    // check arguments
    if(argc == 3) {
        if(iMyRank == 0) {
            LOG(iMyRank, "using user-defined initial load distribution...");    
        } 
        numberOfTasks = atoi(argv[iMyRank + 1]);
    } else { 
        printHelpMessage();
        return 0;     
    }

    // create different size tasks array
    int **mat_size_idx_arr;
    mat_size_idx_arr = new int*[iNumProcs];
    for (int i = 0; i < iNumProcs; i++){
        mat_size_idx_arr[i] = new int[numberOfTasks];
        for (int j = 0; j < numberOfTasks; j++)
            mat_size_idx_arr[i][j] = 0;
    }
    create_diff_task_sizes(iMyRank, numberOfTasks, iNumProcs, mat_size_idx_arr);
    MPI_Barrier(MPI_COMM_WORLD);

    if (iMyRank == 1){
        for (int i = 0; i < numberOfTasks; i++){
            mat_size_idx_arr[iMyRank][i] = mat_size_idx_arr[0][numberOfTasks-i-1];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
	
    // log the info
    std::string msg = "will create "+std::to_string(numberOfTasks)+" tasks";
    LOG(iMyRank, msg.c_str());

    // init the matrices
	double **matrices_a, **matrices_b, **matrices_c;
	matrices_a = new double*[numberOfTasks];
	matrices_b = new double*[numberOfTasks];
	matrices_c = new double*[numberOfTasks];
    int matrixSize[numberOfTasks];

    if(iMyRank == 0) {
        printf("Executing parallel init\n");
    }
    #pragma omp parallel for
	for (int i = 0; i < numberOfTasks; i++) {
        // create random matrix_sizes, matrixSize is set above, just assign again
        // matrixSize[i] = range_size[(rand() % 5)];
        matrixSize[i] = range_size[mat_size_idx_arr[iMyRank][i]];
        // matrixSize[i] = 2048;    // just for testing the tool
        total_load[iMyRank] += matrixSize[i];

 		matrices_a[i] = new double[(long)matrixSize[i]*matrixSize[i]];
    	matrices_b[i] = new double[(long)matrixSize[i]*matrixSize[i]];
    	matrices_c[i] = new double[(long)matrixSize[i]*matrixSize[i]];

        initialize_matrix_test_A(matrices_a[i], matrixSize[i]);
        initialize_matrix_test_A(matrices_b[i], matrixSize[i]);
        initialize_matrix_zero(matrices_c[i], matrixSize[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    fTimeStart = MPI_Wtime();
    #pragma omp parallel
    {
		#pragma omp for
        for(int i = 0; i < numberOfTasks; i++) {
            // printf("R%d - Thread %d --- running --- iter %d\n", iMyRank, omp_get_thread_num(), i);
            double * SPEC_RESTRICT A = matrices_a[i];
            double * SPEC_RESTRICT B = matrices_b[i];
            double * SPEC_RESTRICT C = matrices_c[i];

            // here we need to call library function to add task entry point and parameters by hand
            void* literal_matrix_size   = *(void**)(&matrixSize[i]);
            void* literal_i             = *(void**)(&i);

            // printf("Process %d: num of threads = %d\n", iMyRank, omp_get_num_threads());

            chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[5];
            args[0] = chameleon_map_data_entry_create(A, matrixSize[i]*matrixSize[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[1] = chameleon_map_data_entry_create(B, matrixSize[i]*matrixSize[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[2] = chameleon_map_data_entry_create(C, matrixSize[i]*matrixSize[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
            args[3] = chameleon_map_data_entry_create(literal_matrix_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
            args[4] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

            // create opaque task here
            printf("Task %d belongs R%d, mat_size = %d\n", i, iMyRank, matrixSize[i]);
            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&matrixMatrixKernel, 5, args);
            // printf("R%d - Thread%d: created suscessfully Task %d\n", iMyRank, omp_get_thread_num(), i);

            // add task to the queue
            int32_t res = chameleon_add_task(cur_task);
            // printf("[Debug] R%d - Thread %d: Passed CH_add_task() - Task %d\n", iMyRank, omp_get_thread_num(), i);
            // clean up again
            delete[] args;
            // get the id of the last task added
            // printf("[Debug] R%d - Thread %d call chameleon_get_last_local_id_tasks()\n", iMyRank, omp_get_thread_num());
            TYPE_TASK_ID last_t_id = chameleon_get_last_local_task_id_added();
            // printf("[Debug] R%d - Thread %d: Passed ch_get_last_local_task_id_added()\n", iMyRank, omp_get_thread_num());
        }
        // distribute tasks and execute
	    // printf("[Debug] Call chameleon_distributed_taskwait()\n");
    	int res = chameleon_distributed_taskwait(0);
	    // printf("[Debug] Passed chameleon_distributed_taskwait()\n");
        #pragma omp single
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // get total load on each process
    printf("Total load on R%d: %d\n", iMyRank, total_load[iMyRank]);
    // cham_stats_print_stats();

    // get execution_time for each process
    fTimeEnd = MPI_Wtime();
    wTimeCham = fTimeEnd - fTimeStart;
    if(iMyRank == 0) {
        printf("#R%d: Computations with chameleon took %.5f\n", iMyRank, wTimeCham);
    }
    LOG(iMyRank, "Validation:");

    // check the result of each task
    if(numberOfTasks > 0) {
        for(int t = 0; t <  numberOfTasks; t++) {
            pass &= check_test_matrix(matrices_c[t], matrixSize[t], matrixSize[t]);
        }
        if(pass)
            LOG(iMyRank, "TEST SUCCESS");
        else
            LOG(iMyRank, "TEST FAILED");
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    //deallocate matrices
    for(int i = 0; i < numberOfTasks; i++) {
    	delete[] matrices_a[i];
    	delete[] matrices_b[i];
    	delete[] matrices_c[i];
    }

    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;
    delete[] mat_size_idx_arr;
    MPI_Barrier(MPI_COMM_WORLD);

    #pragma omp parallel
    {
        chameleon_thread_finalize();
    }
    chameleon_finalize();

    MPI_Finalize();
    return 0;
}
