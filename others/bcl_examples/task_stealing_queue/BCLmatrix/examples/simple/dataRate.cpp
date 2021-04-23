#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <string>
#include <cstdlib>
#include <time.h>

#include <random>

double fTimeStart, fTimeEnd;
bool taskStealing = true;
int stolen = 0;
int freed = 0;

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
    int tasks = 0;

    //get the number of tasks to enqueue
    // int rankOffset = 0;
    // for (int i = 2; i < argc; i += 2)
    // {
    //     if (atoi(argv[i + 1]) + rankOffset > BCL::rank() && rankOffset <= BCL::rank())
    //     {
    //         tasks = atoi(argv[i]);
    //     }
    //     rankOffset += atoi(argv[i + 1]);
    // }

    tasks = 20;
    std::cout << "R"+std::to_string(BCL::rank()) << " is creating " << tasks << " tasks" << std::endl;

    std::vector<BCL::CircularQueue<task>> queues;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {
        queues.push_back(BCL::CircularQueue<task>(rank, 100000));
    }

    //create tasks
    for (int it = 0; it < 1; it++)
    {
        for (int i = 0; i < tasks; i++)
        {
            struct task t;
            t.matrix = (double *)malloc(sizeof(double) * matrixSize * matrixSize);
            t.matrix2 = (double *)malloc(sizeof(double) * matrixSize * matrixSize);
            t.matrixSize = matrixSize;
            initialize_matrix_rnd(t.matrix, matrixSize);
            initialize_matrix_rnd(t.matrix2, matrixSize);
            queues[BCL::rank()].push(t, BCL::CircularQueueAL::push);
        }

        BCL::barrier();

        //steal tasks
        if (queues[BCL::rank()].empty())
        {
            fTimeStart = curtime();
            if (steal(&queues))
            {
                fTimeEnd = curtime();
                double elapsed = fTimeEnd - fTimeStart;
                printf("Stolen %d tasks with a rate of %lf MB/s\n", stolen, (matrixSize * matrixSize * 8 * 2 * stolen) / (elapsed * 1000000));
            }
        }

        BCL::barrier();

        sleep(1);
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
    // printf("[%ld]Current: %d\n", BCL::rank(), *it);
    long size = (*queues)[0].size();
    if (size > 1)
    {
        //printf("[%ld]found %d!\n", BCL::rank(),i);
        task t;
        //steals half the tasks
        for (int j = 0; j < size / 2; j++)
        {

            (*queues)[0].pop(t, BCL::CircularQueueAL::pop);
            //printf("[%ld]stealing task %d\n", BCL::rank(), j);
            (*queues)[BCL::rank()].push(t);
        }
        int ownSize = (*queues)[BCL::rank()].size();

        if (ownSize>0)
        {
            stolen += ownSize;
            return true;
        }
    }
    return false;
}
