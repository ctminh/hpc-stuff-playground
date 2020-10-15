#ifndef __STENCIL_H__
#define __STENCIL_H__

#include <stdlib.h>
#include <stdio.h>
#include <starpu.h>

#ifndef STARPU_USE_MPI
#define STARPU_USE_MPI 0
#endif

// if define CUDA
#ifndef __CUDACC__
#if STARPU_USE_MPI && !defined(STARPU_USE_MPI_MASTER_SLAVE)
#include <mpi.h>
#include <starpu_mpi.h>
#endif
#endif

// define error display
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

// define life
#define LIFE

#ifdef LIFE
#define TYPE	unsigned char
extern void life_update(int bz, const TYPE *old, TYPE *newp, int nx, int ny, int nz, int ldy, int ldz, int iter);
#else
#define TYPE	float
#endif

#define K	1   // what is K???

#define NDIRS 2 // what is NDIRs???

/* Split only on the z axis to make things simple */
typedef enum
{
	B = 0,
	T = 1
} direction;

/* description of a domain block */
struct block_description
{
    /* which MPI node should process that block? */
    int mpi_node;
    unsigned perferred_worker;
    unsigned bz;

    /* For each of the following buffers, there are two (0/1) buffers to
	* make new/old switch costless. */
    
    /* This is the computation buffer for this block, it includes
	* neighbours' border to make computation easier */
    TYPE *layers[2];
	starpu_data_handle_t layers_handle[2];

    /* This is the "save" buffer, i.e. a copy of our neighbour's border.
	 * This one is used for CPU/GPU or MPI communication (rather than the
	 * whole domain block) */
	TYPE *boundaries[NDIRS][2];
	starpu_data_handle_t boundaries_handle[NDIRS][2];

	/* Shortcut pointer to the neighbours */
	struct block_description *boundary_blocks[NDIRS];
};

/* define MPI and StarPU tags */
#define TAG_INIT_TASK	((starpu_tag_t)1)
starpu_tag_t TAG_FINISH(int z);
starpu_tag_t TAG_START(int z, int dir);
int MPI_TAG0(int z, int iter, int dir);
int MPI_TAG1(int z, int iter, int dir);

/* define min function */
#define MIN(a,b)	((a)<(b)?(a):(b))

/* define some util functions */
void create_blocks_array(unsigned sizex, unsigned sizey, unsigned sizez, unsigned nbz);
struct block_description *get_block_description(int z);
void assign_blocks_to_mpi_nodes(int world_size);
void allocate_memory_on_node(int rank);
void assign_blocks_to_workers(int rank);

#endif /* __STENCIL_H__ */