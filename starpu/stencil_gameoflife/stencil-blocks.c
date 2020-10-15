#include "stencil.h"
#include <math.h>

/* Manage block and tags allocation */
static struct block_description *blocks;    // an array of blocks
static unsigned sizex, sizey, sizez;
static unsigned nbz;
static unsigned *block_sizes_z;

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
    STARPU_ASSERT(blocks);

    /* what is the size of the different blocks? */
    compute_block_sizes();

    unsigned bz;
    for (bz = 0; bz < nbz; bz++)
    {
        struct block_description * block = get_block_description(bz);
        // which block is it?
        block->bz = bz;

        // for simplicity, we store which are the neighbours blocks
        // printf("\t[create_blocks_array] block %d: B=%d (%d), T=%d (%d)\n", bz, B, ((bz-1+nbz) % nbz), T, ((bz+1) % nbz));
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
void assassign_blocks_to_workers(int rank)
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
            block->perferred_worker = worker_id;
            attributed++;
        }
    }
}