#include "stencil.h"


#ifdef STARPU_QUICK_CHECK
static unsigned niter = 4;
#define SIZE 16
#define NBZ 8
#else
static unsigned niter = 32;
#define SIZE 128
#define NBZ 64
#endif

/* default parameter values */
static unsigned bind_tasks = 0;
static unsigned ticks = 1000;

/* Problem size */
static unsigned sizex = SIZE;
static unsigned sizey = SIZE;
static unsigned sizez = NBZ*SIZE;

/* Number of blocks (scattered over the different MPI processes) */
unsigned nbz = NBZ;

/* Initialization */


int main(int argc, char **argv)
{
    int rank;
    int world_size;
    int ret;

    // STARPU USE MPI
    printf("[CHECK] STARPU_USE_MPI=%d\n", STARPU_USE_MPI);
    #if STARPU_USE_MPI
        int thread_support;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &thread_support)){
            FPRINTF(stderr, "MPI_Init_thread failed\n");
        }

        if (thread_support == MPI_THREAD_FUNNELED)
		    FPRINTF(stderr,"Warning: MPI only has funneled thread support, not serialized, hoping this will work\n");

	    if (thread_support < MPI_THREAD_FUNNELED)
		    FPRINTF(stderr,"Warning: MPI does not have thread support!\n");

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    #else
        rank = 0;
        world_size = 1;
    #endif

    if (rank == 0){
        FPRINTF(stderr, "Running on %d nodes\n", world_size);
		fflush(stderr);
    }

    // init starpu
    printf("1. init StarPU ...");
    ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    return 0;
}