#include "stencil.h"
#include <math.h>

/* Manage block and tags allocation */
static struct block_description *blocks;
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
}