#include "stencil.h"


/* default parameter values */
#ifdef STARPU_QUICK_CHECK
static unsigned niter = 4;
#define SIZE 16
#define NBZ 8
#else
static unsigned niter = 32;
#define SIZE 128
#define NBZ 64
#endif

static unsigned bind_tasks = 0;
static unsigned ticks = 1000;

/* Problem size */
static unsigned sizex = SIZE;
static unsigned sizey = SIZE;
static unsigned sizez = NBZ*SIZE;

/* Number of blocks (scattered over the different MPI processes) */
unsigned nbz = NBZ;

/* Parsing the arguments */
static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0){
			bind_tasks = 1;
		}

		if (strcmp(argv[i], "-nbz") == 0){
			nbz = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-sizex") == 0){
			sizex = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-sizey") == 0){
			sizey = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-sizez") == 0){
			sizez = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-niter") == 0){
			niter = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-ticks") == 0){
			ticks = atoi(argv[++i]);
		}

		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0){
			 fprintf(stderr, "Usage : %s [options...]\n", argv[0]);
			 fprintf(stderr, "\n");
			 fprintf(stderr, "Options:\n");
			 fprintf(stderr, "-b    bind tasks on CPUs/GPUs\n");
			 fprintf(stderr, "-nbz <n>  Number of blocks on Z axis (%u by default)\n", nbz);
			 fprintf(stderr, "-size[xyz] <size>	Domain size on x/y/z axis (%ux%ux%u by default)\n", sizex, sizey, sizez);
			 fprintf(stderr, "-niter <n>    Number of iterations (%u by default)\n", niter);
			 fprintf(stderr, "-ticks <t>	How often to put ticks in the output (ms, %u by default)\n", ticks);
			 exit(0);
		}
	}
}

/* Initialization */
static void init_problem(int argc, char **argv, int rank, int world_size)
{
	// parse the arguments if yes
    printf("\t[init_problem] parsing arguments\n");
    parse_args(argc, argv);

	// create block_arrays
    printf("\t[init_problem] creating block_arrays\n");
	printf("\t size_x=%d, size_y=%d, size_z=%d, num_blocks=%d\n", sizex, sizey, sizez, nbz);
    create_blocks_array(sizex, sizey, sizez, nbz);

	// assign blocks to MPI nodes
	assign_blocks_to_mpi_nodes(world_size);

	// assign blocks to workers
	assign_blocks_to_workers(rank);

	// allocate the different mem blocks, if used by the MPI process
	allocate_memory_on_node(rank);

	// display mem usage
	display_memory_consumption(rank);


}


/* Main body */
double start;
double begin, end;
double timing; 

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
    printf("1. init StarPU ...\n");
    ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    // init starpu_mpi if we use mpi
    #if STARPU_USE_MPI
    ret = starpu_mpi_init(NULL, NULL, 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_mpi_init");
    #endif

    // init the problem
    printf("2. init the problem ...\n");
    init_problem(argc, argv, rank, world_size);

	// check the assignment of blocks to mpi_nodes
	unsigned b_idx;
	for (b_idx = 0; b_idx < nbz; b_idx++){
		struct block_description *block = get_block_description(b_idx);
		printf("\t[main] block %d -> mpi_rank_%d, preferred_worker_%d\n", block->bz, block->mpi_node, block->perferred_worker);
	}

    return 0;
}