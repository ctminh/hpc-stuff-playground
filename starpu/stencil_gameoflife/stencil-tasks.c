#include "stencil.h"

/*
 * Schedule tasks for updates and saves
 */

/*
 * NB: iter = 0: initialization phase, TAG_U(z, 0) = TAG_INIT
 * dir is -1 or +1.
 */

#if 0
    #define DEBUG(fmt, ...) fprintf(stderr,fmt,##__VA_ARGS__)
#else
    #define DEBUG(fmt, ...)
#endif

/* Define starpu codelet */
void null_func(void *descr[], void *arg)
{
    (void) descr;
    (void) arg;
}

static double null_cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 0.000001;
}

static struct starpu_perfmodel null_model =
{
	.type = STARPU_COMMON,
	.cost_function = null_cost_function,
	.symbol = "null"
};

static struct starpu_codelet null = 
{
    .modes = {STARPU_W, STARPU_W},
    .cpu_funcs = {null_func},
    .cpu_funcs_name = {"null_func"},
    .cuda_funcs = {null_func},
    .opencl_funcs = {null_func},
    .nbuffers = 2,
    .model = &null_model,
    .name = "start"
};


/* Create start tasks??? */
void create_start_task(int z, int dir)
{
    // ------------------------ begin VT -----------------------------
    #ifdef TRACE
    int event_c_starttask = -1;
    char eventtag_c_starttask[12] = "c_start_task";
    int itac_err = VT_funcdef(eventtag_c_starttask, VT_NOCLASS, &event_c_starttask);
    VT_BEGIN_CONSTRAINED(event_c_starttask);
    #endif
    // ---------------------------------------------------------------

    /* Dumb task depending on the init task and simulating writing the
	   neighbour buffers, to avoid communications and computation running
	   before we start measuring time */
    struct starpu_task *wait_init = starpu_task_create();
    struct block_description *descr = get_block_description(z);
    starpu_tag_t tag_init = TAG_INIT_TASK;
    wait_init->cl = &null;
    wait_init->use_tag = 1;
    wait_init->tag_id = TAG_START(z, dir);
    wait_init->handles[0] = descr->boundaries_handle[(1 + dir) / 2][0];
    wait_init->handles[1] = descr->boundaries_handle[(1 + dir) / 2][1];
    starpu_tag_declare_deps_array(wait_init->tag_id, 1, &tag_init);

    int ret = starpu_task_submit(wait_init);
    if (ret)
    {
        FPRINTF(stderr, "Could not submit task initial wait: %d\n", ret);
        if (ret == -ENODEV){
            exit(77);
        }
        STARPU_ABORT();
    }

    // ------------------------ end VT -------------------------------
    #ifdef TRACE
    VT_END_W_CONSTRAINED(event_c_starttask);
    #endif
    // ---------------------------------------------------------------
    
}


/*
 * Create all the tasks
 */
void create_tasks(int rank)
{
    // ------------------------ begin VT -----------------------------
    #ifdef TRACE
    int event_c_task = -1;
    char eventtag_c_task[12] = "c_task";
    int itac_err = VT_funcdef(eventtag_c_task, VT_NOCLASS, &event_c_task);
    VT_BEGIN_CONSTRAINED(event_c_task);
    #endif
    // ---------------------------------------------------------------

    int iter;
    int bz;
    int niter = get_niter();
    int nbz = get_nbz();
    for (bz = 0; bz < nbz; bz++)
    {
        if ((get_block_mpi_node(bz) == rank) || (get_block_mpi_node(bz+1) == rank))
            create_start_task(bz, +1);
        if ((get_block_mpi_node(bz) == rank) || (get_block_mpi_node(bz-1) == rank))
            create_start_task(bz, -1);
    }

    // ------------------------ end VT -------------------------------
    #ifdef TRACE
    VT_END_W_CONSTRAINED(event_c_task);
    #endif
    // ---------------------------------------------------------------
}