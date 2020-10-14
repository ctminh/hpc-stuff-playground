#include "stencil.h"
#include <math.h>

/* Manage block and tags allocation */
static struct block_description *blocks;    // an array of blocks
static unsigned sizex, sizey, sizez;
static unsigned nbz;
static unsigned *block_sizes_z;

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
        printf("\t[create_blocks_array] block %d: B=%d (%d), T=%d (%d)\n", bz, B, ((bz-1+nbz) % nbz), T, ((bz+1) % nbz));
        block->boundary_blocks[B] = get_block_description((bz-1+nbz) % nbz);
        block->boundary_blocks[T] = get_block_description((bz+1) % nbz);
    }
}