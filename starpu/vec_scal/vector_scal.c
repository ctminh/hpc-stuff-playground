#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NX 20480

// declare task functions
extern void scal_cpu_func(void *buffers[], void *_args);
extern void scal_cuda_func(void *buffers[], void *_args);

/* don't know yet for what? */
static struct starpu_perfmodel vector_scal_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scal"
};

/* don't know yet for what? */
static struct starpu_perfmodel vector_scal_energy_model =
{
	.type = STARPU_HISTORY_BASED,
	.symbol = "vector_scal_energy"
};

/* define a codelet */
static struct starpu_codelet cl =
{
    // options to run the program
    .where = STARPU_CPU | STARPU_CUDA,
    // CPU implementation
    .cpu_funcs = {scal_cpu_func},
    .cpu_funcs_name = {"scal_cpu_func"},
    // CUDA implementation
    #ifdef STARPU_USE_CUDA
    .cuda_funcs = {scal_cuda_func},
    #endif


    // data
    .nbuffers = 1,
    .modes = {STARPU_RW},
	.model = &vector_scal_model,
	.energy_model = &vector_scal_energy_model
};

/////////////////////////////////////////////
/* main function */
int main(int argc, char *argv[])
{
    /* We consider a vector of float that is initialized just as any of C data */
    printf("1. Init the vector\n");
	float vector[NX];   // init the vector
	unsigned i;
	for (i = 0; i < NX; i++)
        vector[i] = (i + 1.0f);

    fprintf(stderr, "BEFORE: First element was %fnn\n", vector[0]);

    /* init StarPU */
    printf("2. Init starPU\n");
    int ret = starpu_init(NULL);
    if (ret == -ENODEV)
		goto enodev;

    /* create starpu tasks */
    printf("3. Create starpu task\n");
    starpu_data_handle_t vector_handle;
    starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

    float factor = 3.14;
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->handles[0] = vector_handle;
    task->cl_arg = &factor;
    task->cl_arg_size = sizeof(factor);

    // submit task
    printf("4. Submit starpu task\n");
    ret = starpu_task_submit(task);

    // unregister data
    printf("5. Unregister data and free memory - Finish the program\n");
    starpu_data_unregister(vector_handle);
	starpu_memory_unpin(vector, sizeof(vector));
    starpu_shutdown();

    fprintf(stderr, "AFTER First element is %fnn\n", vector[0]);

    return (ret ? EXIT_SUCCESS : EXIT_FAILURE);

    enodev:
		return 77;
}