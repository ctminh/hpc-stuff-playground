#include <starpu.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NX 2048000

/* define task function */
void scal_cpu_func(void *buffers[], void *cl_arg)
{
	unsigned i;
	float *factor = cl_arg;

	/* length of the vector */
	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);

	/* get a pointer to the local copy of the vector : note that we have to
	 * cast it in (float *) since a vector could contain any type of
	 * elements so that the .ptr field is actually a uintptr_t */
	float *val = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);

	/* scale the vector */
	for (i = 0; i < n; i++)
		val[i] *= *factor;
}

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
    // CPU implementation
    .cpu_funcs = {scal_cpu_func},
    .cpu_funcs_name = {"scal_cpu_func"},

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
	float vector[NX];   // init the vector
	unsigned i;
	for (i = 0; i < NX; i++)
        vector[i] = (i+1.0f);

    /* init StarPU */
    int ret = starpu_init(NULL);
    if (ret == -ENODEV)
		goto enodev;

    /* create starpu tasks */
    starpu_data_handle_t vector_handle;
    starpu_vector_data_register(&vector_handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

    float factor = 3.14;
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->handles[0] = vector_handle;
    task->cl_arg = &factor;
    task->cl_arg_size = sizeof(factor);

    // submit task
    ret = starpu_task_submit(task);

    // unregister data
    starpu_data_unregister(vector_handle);
	starpu_memory_unpin(vector, sizeof(vector));
    starpu_shutdown();

    return (ret ? EXIT_SUCCESS : EXIT_FAILURE);

    enodev:
		return 77;
}