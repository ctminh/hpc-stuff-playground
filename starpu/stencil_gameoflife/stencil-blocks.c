#include "stencil.h"
#include <math.h>

/* Manage block and tags allocation */
static struct block_description *blocks;    // an array of blocks
static unsigned sizex, sizey, sizez;        // blobal vars about x, y, z
static unsigned nbz;                        // global var about num of blocks
static unsigned *block_sizes_z;             // an array containing sizes of all the blocks
static size_t allocated = 0;


/*
 *	Tags for various codelet completion
 */

/*
 * common tag format:
 */
static starpu_tag_t tag_common(int z, int dir, int type)
{
	return (((((starpu_tag_t)type) << 4) | ((dir+1)/2)) << 32)|(starpu_tag_t)z;
}

/* Completion of last update tasks */
starpu_tag_t TAG_FINISH(int z)
{
	z = (z + nbz)%nbz;

	starpu_tag_t tag = tag_common(z, 0, 1);
	return tag;
}

/* Completion of the save codelet for MPI send/recv */
starpu_tag_t TAG_START(int z, int dir)
{
	z = (z + nbz)%nbz;

	starpu_tag_t tag = tag_common(z, dir, 2);
	return tag;
}

/*
 * common MPI tag format:
 * iter is actually not needed for coherency, but it makes debugging easier
 */
static int mpi_tag_common(int z, int iter, int dir, int buffer)
{
	return (((((iter << 12)|z)<<4) | ((1+dir)/2))<<4)|buffer;
}

int MPI_TAG0(int z, int iter, int dir)
{
	z = (z + nbz)%nbz;
	int tag = mpi_tag_common(z, iter, dir, 0);

	return tag;
}

int MPI_TAG1(int z, int iter, int dir)
{
	z = (z + nbz)%nbz;
	int tag = mpi_tag_common(z, iter, dir, 1);

	return tag;
}


/* Compute the size of the different blocks */
static void compute_block_sizes(void)
{
	block_sizes_z = (unsigned *) malloc(nbz*sizeof(unsigned));
	STARPU_ASSERT(block_sizes_z);

	/* Perhaps the last chunk is smaller */
	unsigned default_block_size = (sizez + nbz - 1) / nbz;
	unsigned remaining = sizez;

	unsigned b;
	for (b = 0; b < nbz; b++)
	{
        // store sizes of other blocks to this golbal arr
		block_sizes_z[b] = MIN(default_block_size, remaining);
		remaining -= block_sizes_z[b];
	}

	STARPU_ASSERT(remaining == 0);
}

/* Get block description */
struct block_description *get_block_description(int z)
{
    z = (z + nbz) % nbz;
    STARPU_ASSERT(&blocks[z]);
    return &blocks[z];
}

/* Create blocks-arrays */
void create_blocks_array(unsigned _sizex, unsigned _sizey, unsigned _sizez, unsigned _nbz)
{
    /* store the parameters */
    nbz = _nbz;
    sizex = _sizex;
    sizey = _sizey;
    sizez = _sizez;

    /* create a grid of block descriptors */
    blocks = (struct block_description *) calloc(nbz, sizeof(struct block_description));

    /* what is the size of the different blocks? */
    compute_block_sizes();

    unsigned bz;
    for (bz = 0; bz < nbz; bz++)
    {
        /* point to the block with the order is bz = 0, 1, ... */
        struct block_description *block = get_block_description(bz);

        /* assign the id for each block */
        block->bz = bz;

        /* for simplicity, we create the pointer for neighbours blocks */
        block->boundary_blocks[B] = get_block_description((bz-1+nbz) % nbz);
        block->boundary_blocks[T] = get_block_description((bz+1) % nbz);
    }
}

/* Assign blocks to MPI nodes */
void assign_blocks_to_mpi_nodes(int world_size)
{
    unsigned nzblocks_per_process = (nbz + world_size - 1) / world_size;
    unsigned bz;
    for (bz = 0; bz < nbz; bz++){
        struct block_description *block = get_block_description(bz);
        block->mpi_node = bz / nzblocks_per_process;
    }
}

/* Assign blocks to workers */
void assign_blocks_to_workers(int rank)
{
    unsigned bz;

    /* NB: perhaps we could count a GPU as multiple workers */
	/*     how many workers are there? */
	/*     unsigned nworkers = starpu_worker_get_count(); */
	/*     how many blocks are on that MPI node? */
        //	unsigned nblocks = 0;
        //	for (bz = 0; bz < nbz; bz++)
        //	{
        //		struct block_description *block =
        //				get_block_description(bz);
        //
        //		if (block->mpi_node == rank)
        //			nblocks++;
        //	}
	/*      how many blocks per worker? */
	/*      unsigned nblocks_per_worker = (nblocks + nworkers - 1)/nworkers; */

    /* we now attribute up to nblocks_per_worker blocks per workers */
    unsigned attributed = 0;
    for (bz = 0; bz < nbz; bz++){
        struct block_description *block = get_block_description(bz);
        if (block->mpi_node == rank){
            unsigned worker_id;
            // manage initial block distribution between CPU and GPU
            #if 0   // GPUs then CPUs
                #if 1
                    if (attributed < 3*18)
                        worker_id = attributed / 18;
                    else
                        worker_id = 3 + (attributed - 3*18) / 2;
                #else
                    if ((attributed % 20) <= 1)
                        workerid = 3 + attributed / 20;
                    else if (attributed < 60)
                        workerid = attributed / 20;
                    else
                        workerid = (attributed - 60) / 2 + 6;
                #endif
            #else   // only GPUs
                worker_id = (attributed / 21) % 3;
            #endif

            // it means = attributed / nblocks_per_worker;
            block->preferred_worker = worker_id;
            attributed++;
        }
    }
}

/* Allocate blocks on node */
void allocate_block_on_node(starpu_data_handle_t *handleptr, unsigned bz, TYPE **ptr, unsigned nx, unsigned ny, unsigned nz)
{
    int ret;
    size_t block_size = nx * ny * nz * sizeof(TYPE);

    /* allocate memory */
    #if 1
        ret = starpu_malloc_flags((void **)ptr, block_size, STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
        STARPU_ASSERT(ret == 0);
    #else
        *ptr = malloc(block_size);
        STARPU_ASSERT(*ptr);
    #endif

    allocated += block_size;

    /* register it to StarPU */
    starpu_block_data_register(handleptr, STARPU_MAIN_RAM, (uintptr_t) *ptr, nx, nx*ny, nx, ny, nz, sizeof(TYPE));
    starpu_data_set_coordinates(*handleptr, 1, bz);
}

/* Allocate the different mem blocks on node */
void allocate_memory_on_node(int rank)
{
    unsigned bz;    // block
    for (bz = 0; bz < nbz; bz++){
        struct block_description *block = get_block_description(bz);
        int node = block->mpi_node;
        // printf("[allocate_mem_on_node] block %d: rank %d\n", bz, node);

        // main blocks
        if (node == rank){
            unsigned size_bz = block_sizes_z[bz];
            // printf("[allocate_mem_on_rank_%d] main block %d: size = %d, block->layers[0] = %p\n", node, bz, size_bz, &block->layers[0]);
            allocate_block_on_node(&block->layers_handle[0], bz, &block->layers[0],
                                        (sizex + 2*K),
                                        (sizey + 2*K),
                                        (size_bz + 2*K));
            // printf("[allocate_mem_on_rank_%d] main block %d: size = %d, block->layers[1] = %p\n", node, bz, size_bz, &block->layers[1]);
            allocate_block_on_node(&block->layers_handle[1], bz, &block->layers[1],
                                        (sizex + 2*K),
                                        (sizey + 2*K),
                                        (size_bz + 2*K));
        }

        // boundary blocks: top
        int top_node = block->boundary_blocks[T]->mpi_node;
        if ((node == rank) || (top_node == rank))
        {
            // printf("[allocate_mem_on_rank_%d/top_rank_%d] top block %d: block->boundaries[T][0] %p\n", node, top_node, bz, &block->boundaries[T][0]);
            allocate_block_on_node(&block->boundaries_handle[T][0], bz, &block->boundaries[T][0],
                                        (sizex + 2*K), (sizey + 2*K), K);
            // printf("[allocate_mem_on_rank_%d/top_rank_%d] top block %d: block->boundaries[T][1] %p\n", node, top_node, bz, &block->boundaries[T][1]);
            allocate_block_on_node(&block->boundaries_handle[T][1], bz, &block->boundaries[T][1],
                                        (sizex + 2*K), (sizey + 2*K), K);
        }

        // boundary blocks: bottom
        int bottom_node = block->boundary_blocks[B]->mpi_node;
        if ((node == rank) || (bottom_node == rank))
        {
            // printf("[allocate_mem_on_rank_%d/bot_rank_%d] bot block %d: block->boundaries[B][0] %p\n", node, bottom_node, bz, &block->boundaries[B][0]);
            allocate_block_on_node(&block->boundaries_handle[B][0], bz, &block->boundaries[B][0],
                                        (sizex + 2*K), (sizey + 2*K), K);
            // printf("[allocate_mem_on_rank_%d/bot_rank_%d] bot block %d: block->boundaries[B][1] %p\n", node, bottom_node, bz, &block->boundaries[B][1]);
            allocate_block_on_node(&block->boundaries_handle[B][1], bz, &block->boundaries[B][1],
                                        (sizex + 2*K), (sizey + 2*K), K);
        }
    }
}

/* Display memory usage */
void display_memory_consumption(int rank)
{
	FPRINTF(stderr, "%lu bytes of memory were allocated on Rank %d\n", (unsigned long) allocated, rank);
}


/* Get block from MPI node/rank */
int get_block_mpi_node(int z)
{
    z = (z + nbz) % nbz;
    return blocks[z].mpi_node;
}

/* Get block size */
unsigned get_block_size(int bz)
{
	return block_sizes_z[bz];
}


/* Free block arrays */
void free_blocks_array()
{
	free(blocks);
	free(block_sizes_z);
}

/* Free block on node */
static void free_block_on_node(starpu_data_handle_t handleptr, unsigned nx, unsigned ny, unsigned nz)
{
	void *ptr = (void *) starpu_block_get_local_ptr(handleptr);
	size_t block_size = nx*ny*nz*sizeof(TYPE);
	starpu_data_unregister(handleptr);
	starpu_free_flags(ptr, block_size, STARPU_MALLOC_PINNED|STARPU_MALLOC_SIMULATION_FOLDED);
}

void free_memory_on_node(int rank)
{
	unsigned bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);

		int node = block->mpi_node;

		/* Main blocks */
		if (node == rank)
		{
			free_block_on_node(block->layers_handle[0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->layers_handle[1], (sizex + 2*K), (sizey + 2*K), K);
		}

		/* Boundary blocks : Top */
		int top_node = block->boundary_blocks[T]->mpi_node;
		if ((node == rank) || (top_node == rank))
		{
			free_block_on_node(block->boundaries_handle[T][0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->boundaries_handle[T][1], (sizex + 2*K), (sizey + 2*K), K);
		}

		/* Boundary blocks : Bottom */
		int bottom_node = block->boundary_blocks[B]->mpi_node;
		if ((node == rank) || (bottom_node == rank))
		{
			free_block_on_node(block->boundaries_handle[B][0], (sizex + 2*K), (sizey + 2*K), K);
			free_block_on_node(block->boundaries_handle[B][1], (sizex + 2*K), (sizey + 2*K), K);
		}
	}
}

/* check how many cells are alive */
void check(int rank)
{
	unsigned bz;
	for (bz = 0; bz < nbz; bz++)
	{
		struct block_description *block = get_block_description(bz);

		int node = block->mpi_node;

		/* Main blocks */
		if (node == rank)
		{
        #ifdef LIFE
			unsigned size_bz = block_sizes_z[bz];
			unsigned x, y, z;
			unsigned sum = 0;
			for (x = 0; x < sizex; x++)
				for (y = 0; y < sizey; y++)
					for (z = 0; z < size_bz; z++)
						sum += block->layers[0][(K+x)+(K+y)*(sizex + 2*K)+(K+z)*(sizex+2*K)*(sizey+2*K)];
			printf("block %u got %u/%u alive\n", bz, sum, sizex*sizey*size_bz);
        #endif
		}
	}
}