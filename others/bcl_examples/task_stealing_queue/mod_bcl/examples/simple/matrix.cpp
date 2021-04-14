#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <string>
#include <cstdlib>
#include <time.h>

#include <random>

double fTimeStart, fTimeEnd;
bool taskStealing = false;
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
    int rankOffset = 0;
    for (int i = 2; i < argc; i += 2)
    {
        if (atoi(argv[i + 1]) + rankOffset > BCL::rank() && rankOffset <= BCL::rank())
        {
            tasks = atoi(argv[i]);
        }
        rankOffset += atoi(argv[i + 1]);
    }

    std::vector<BCL::CircularQueue<task>> queues;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {
        queues.push_back(BCL::CircularQueue<task>(rank, 100000));
    }

    if (BCL::rank() == 0)
        fTimeStart = curtime();

    //create tasks
    for (int it = 0; it < 5; it++)
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
        // for (int i = 0; i < BCL::nprocs(); i++)
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
            bool success = queues[BCL::rank()].pop(t);
            if (success)
            {
                double *result = new double[matrixSize * matrixSize];
                multiply(t.matrix, t.matrix2, result, matrixSize);
                free(t.matrix);
                free(t.matrix2);
                delete[] result;
                // freed += 2;
                // if(BCL::rank()==0)
                //     printf("[%ld]Freed matrix: %d\n", BCL::rank(),freed);
            }
        }
        BCL::barrier();
    }

    if (BCL::rank() == 0)
    {
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
    std::srand(unsigned(std::time(0)));
    std::vector<int> ranks;
    for (int i = 0; i < BCL::nprocs(); ++i)
    {
        if (i != BCL::rank())
            ranks.push_back(i);
    }

    std::random_shuffle(ranks.begin(), ranks.end());

    int i = (BCL::rank() + 1) % BCL::nprocs();
    //iterates through ranks and tests if they have tasks left
    for (std::vector<int>::iterator it = ranks.begin(); it != ranks.end(); ++it)
    {
        // printf("[%ld]Current: %d\n", BCL::rank(), *it);
        if ((*queues)[*it].size() > 1)
        {
            //printf("[%ld]found %d!\n", BCL::rank(),i);
            task t;
            //steals half the tasks
            for (int j = 0; j < (*queues)[*it].size() / 2; j++)
            {

                (*queues)[*it].pop(t, BCL::CircularQueueAL::pop);
                //printf("[%ld]stealing task %d\n", BCL::rank(), j);
                (*queues)[BCL::rank()].push(t);
            }

            if (!(*queues)[BCL::rank()].empty())
            {
                printf("[%ld]Successfully stolen %ld tasks!\n", BCL::rank(), (*queues)[BCL::rank()].size());
                return true;
            }
        }
    }
    return false;
}
