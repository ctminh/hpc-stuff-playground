#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <string>
#include <cstdlib>
#include <time.h>

#include <random>

double fTimeStart, fTimeEnd;
bool taskStealing = true;

void multiply(double *matrix, double *matrix2, double *result, int matrixSize);
void initialize_matrix_rnd(double *mat, int matrixSize);
bool steal(std::vector<BCL::CircularQueue<task>> *queues);

static inline double curtime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    BCL::init();

    int matrixSize = atoi(argv[1]);
    int tasks = atoi(argv[BCL::rank() + 2]);

    std::vector<BCL::CircularQueue<task>> queues;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {

        queues.push_back(BCL::CircularQueue<task>(rank, 100000));
    }

    if (BCL::rank() == 0)
        fTimeStart = curtime();
        // fTimeStart = MPI_Wtime();

    //create tasks
    for (int i = 0; i < tasks; i++)
    {
        struct task t;
        t.matrix = (double *)malloc(sizeof(double) * matrixSize * matrixSize);
        t.matrix2 = (double *)malloc(sizeof(double) * matrixSize * matrixSize);
        t.matrixSize = matrixSize;
        initialize_matrix_rnd(t.matrix, matrixSize);
        initialize_matrix_rnd(t.matrix2, matrixSize);
        queues[BCL::rank()].push(t);
    }

    BCL::barrier();
    //for (int i = 0; i < BCL::nprocs(); i++)
    //    printf("[%ld]Rank %d:%ld\n", BCL::rank(), i, queues[i].size());

    //solve tasks
    while (true)
    {
        //steal tasks if activated
        if (queues[BCL::rank()].empty())
        {
            if (!taskStealing)
                break;
            if (!steal(&queues))
            {
                break;
            }
        }

        task t;
        queues[BCL::rank()].pop(t);
        double *result = (double *)malloc(sizeof(double) * matrixSize * matrixSize);
        multiply(t.matrix, t.matrix2, result, matrixSize);
        if(BCL::rank() == 3)
            printf("%ld\n",(queues)[BCL::rank()].size());
        // printf("[%ld]Freed %d\n", BCL::rank(), t.matrixSize*t.matrixSize*2);
        free(t.matrix);
        free(t.matrix2);
        free(result);
        // int flag;
        // MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }

    BCL::barrier();

    if (BCL::rank() == 0){
        fTimeEnd = curtime();
        double elapsed = fTimeEnd - fTimeStart;
        printf("%lf\n", elapsed);
    }

    BCL::finalize();
}

void initialize_matrix_rnd(double *mat, int matrixSize)
{
    double lower_bound = 0;
    double upper_bound = 10;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        mat[i] = unif(re);
    }
}

void multiply(double *matrix, double *matrix2, double *result, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i += 1)
    {
        double value = 0;
        int k = i % matrixSize;
        for (int j = (i / matrixSize) * matrixSize; j < (i / matrixSize) * matrixSize + matrixSize; j++)
        {
            value = value + matrix[j] * matrix2[k];
            k += matrixSize;
        }
        result[i] = value;
    }
}

bool steal(std::vector<BCL::CircularQueue<task>> *queues)
{
    int i = (BCL::rank() + 1) % BCL::nprocs();
    //iterates through ranks and tests if they have tasks left
    while (i != BCL::rank())
    { //printf("[%ld]testing %d\n",BCL::rank(), i);
        //printf("[%ld]Trying to steal from rank %d: size: %d\n", BCL::rank(), i, (*queues)[i].size());
        if ((*queues)[i].size() > 1)
        {
            //printf("[%ld]found %d!\n", BCL::rank(),i);
            task t;
            //steals half the tasks
            for (int j = 0; j < (*queues)[i].size() / 2; j++)
            {

                (*queues)[i].pop(t);
                // printf("stealing task %d\n",j);
                (*queues)[BCL::rank()].push(t);
            }

            if (!(*queues)[BCL::rank()].empty())
            {
                printf("Successfully stolen %ld tasks!\n", (*queues)[BCL::rank()].size());
                return true;
            }
        }
        else
        {
            i = (i + 1) % BCL::nprocs();
        }
    }
    return false;
}
