#include <assert.h>
#include <mpi.h>
#include "chameleon.h"
#include <cstdlib>
#include <cstdio>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h>
#include <sys/syscall.h>
#include <atomic>
#include <ctime>
#include <math.h>
#include <mkl.h>

#define LOG(rank, str) printf("#R%d: %s\n", rank, str)

#define SPEC_RESTRICT __restrict__

// define constants for Jacobi tasks
#define MAX_ITERATIONS 20000
#define CONVERGENCE_THRESHOLD  0.01
#define PRIME1 293
#define PRIME2 719

/* ////////////////////// Matrix-Multiplication Task /////////////////////////////
/////////////////////////////////////////////////////////////////////////////// */

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

/* matrix-mul kernel */
void matrixMatrixKernel(double * SPEC_RESTRICT A, double * SPEC_RESTRICT B, double * SPEC_RESTRICT C, int matrixSize, int i) {
    compute_matrix_matrix(A, B, C, matrixSize);
}

/* function to create mat-mul tasks */
void create_mat_mul_tasks(int number_of_tasks, double **matrices_a, double **matrices_b, double **matrices_c, int *size_arr)
{
    #pragma omp parallel
    {
		#pragma omp for
        for(int i = 0; i < number_of_tasks; i++) {
            // printf("R%d - Thread %d --- running --- iter %d\n", iMyRank, omp_get_thread_num(), i);
            double * SPEC_RESTRICT A = matrices_a[i];
            double * SPEC_RESTRICT B = matrices_b[i];
            double * SPEC_RESTRICT C = matrices_c[i];

            // here we need to call library function to add task entry point and parameters by hand
            void* literal_matrix_size   = *(void**)(&size_arr[i]);
            void* literal_i             = *(void**)(&i);

            // printf("Process %d - Thread %d create Task %d\n", iMyRank, omp_get_thread_num(), i);

            chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[5];
            args[0] = chameleon_map_data_entry_create(A, size_arr[i]*size_arr[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[1] = chameleon_map_data_entry_create(B, size_arr[i]*size_arr[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[2] = chameleon_map_data_entry_create(C, size_arr[i]*size_arr[i]*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
            args[3] = chameleon_map_data_entry_create(literal_matrix_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
            args[4] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

            // create opaque task here
            cham_migratable_task_t *cur_task = chameleon_create_task((void *)&matrixMatrixKernel, 5, args);

            // add task to the queue
            int32_t res = chameleon_add_task(cur_task);
            
            delete[] args;
            // get the id of the last task added
            TYPE_TASK_ID last_t_id = chameleon_get_last_local_task_id_added();
        }
    }
}

/* ////////////////////// Jacobi-2D Task /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////// */

/* init jacobi data */
void init_jacobi_data(double *A, double *b, double *x, int mat_size)
{
    srand(0);
    int N = mat_size;
    for (int row = 0; row < N; row++){
        double rowsum = 0.0;
        for (int col = 0; col < N; col++){
            double value = rand()/(double)RAND_MAX;
            A[row+ col*N] = value;
            rowsum += value;
        }
        A[row + row*N] += rowsum;
        b[row] = rand()/(double)RAND_MAX;
        x[row] = 0.0;
    }
}

int solver(double * SPEC_RESTRICT A, double * SPEC_RESTRICT b, double * SPEC_RESTRICT x, double * SPEC_RESTRICT xtmp, int size){
    int itr;
    int row, col;
    double dot;
    double diff;
    double sqdiff;
    double *ptrtmp;
    int N =  size;

    /* For example: 
    A00x0   +   A01x1   +   A02x2   =   b0
    A10x0   +   A11x1   +   A12x2   =   b1
    A20x0   +   A21x1   +   A22x2   =   b2
    --------------------------------------
    x0_0 = 0, x1_0 = 0, x2_0 = 0
    --------------------------------------
    x0_1 = [b0 - (A01x1_0 + A02x2_0)] / A00
    x1_1 = [b1 - (A10x0_0 + A12x2_0)] / A11
    ...
    --------------------------------------*/

    itr = 0;
    do{
        for (row = 0; row < N; row++){
            dot = 0.0;
            for (col = 0; col < N; col++){
                if (row != col)
                    // caculate the dot
                    dot += A[row + col * N]  * x[col];
            }
            // update x from x0
            xtmp[row] = (b[row] - dot) / A[row + row * N];
        }

        // swap pointers to update new values for x
        ptrtmp = x;
        x = xtmp;
        xtmp = ptrtmp;

        // check the convergence
        sqdiff = 0.0;
        for (row = 0; row < N; row++){
            diff = xtmp[row] - x[row];
            sqdiff += diff * diff;
        }

        // increase the iteration
        itr++;
    } while ((itr < MAX_ITERATIONS) && (sqrt(sqdiff) > CONVERGENCE_THRESHOLD));

    return itr;
}

/* jacobi kernel */
void jacobi_kernel(double * SPEC_RESTRICT cham_A, double * SPEC_RESTRICT cham_b, double * SPEC_RESTRICT cham_x, double * SPEC_RESTRICT cham_xtmp, int size, int i){
    int num_itr = solver(cham_A, cham_b, cham_x, cham_xtmp, size);
}

/* create jacobi tasks */
void create_jacobi_tasks(int numberOfTasks, double **jaco_A, double **jaco_b, double **jaco_x, double **jaco_xtmp, int mat_size)
{
    #pragma omp parallel
    {
        #pragma omp for
    	for(int i = 0; i < numberOfTasks; i++) {
            double * SPEC_RESTRICT cham_A = jaco_A[i];
            double * SPEC_RESTRICT cham_b = jaco_b[i];
            double * SPEC_RESTRICT cham_x = jaco_x[i];
            double * SPEC_RESTRICT cham_xtmp = jaco_xtmp[i];
            void* literal_mat_size   = *(void**)(&mat_size);
            void* literal_i          = *(void**)(&i);

            // create data with chameleon
            chameleon_map_data_entry_t *args = new chameleon_map_data_entry_t[6];
            args[0] = chameleon_map_data_entry_create(cham_A, mat_size*mat_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[1] = chameleon_map_data_entry_create(cham_b, mat_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
            args[2] = chameleon_map_data_entry_create(cham_x, mat_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
            args[3] = chameleon_map_data_entry_create(cham_xtmp, mat_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
            args[4] = chameleon_map_data_entry_create(literal_mat_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
            args[5] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
            
            // create tasks
            cham_migratable_task_t *task = chameleon_create_task((void *)&jacobi_kernel, 6, args);
            // add task to the queue
            int32_t res = chameleon_add_task(task);
            // delete parameter arr
            delete[] args;
        }
    }
}

/* ////////////////////// Cholesky Task /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////// */


/* ////////////////////// Util-Functions /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////// */

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
