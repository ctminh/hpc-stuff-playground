#include <starpu.h>
#include <assert.h>
#include <unistd.h>

/* define print-debug */
#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

/* define some variables */
static unsigned n_iter = 50;

/* define task */
void sleep_codelet(STARPU_ATTRIBUTE_UNUSED void *descr[],
			STARPU_ATTRIBUTE_UNUSED void *_args)
{
	usleep(2000);
}

/* define a codelet */
struct starpu_codelet cl =
{
    .cpu_funcs = {sleep_codelet},
    .cpu_funcs_name = {"sleep_codelet"},
    .cuda_funcs = {sleep_codelet},
    .opencl_funcs = {sleep_codelet},
    .nbuffers = 0,
    .name = "sleep"
};

int main(int argc, char *argv[])
{
    int ret;
    if (argc == 2)  // get num of iterations
        n_iter = atoi(argv[1]);
    
    // init startpu
    ret = starpu_init(NULL);
    if (ret == -ENODEV)
        return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    /* enable profiling */
    starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

    /* starpu declares tasks */
    struct starpu_task **tasks = (struct starpu_task **) malloc(n_iter*sizeof(struct starpu_task *));
    assert(tasks);
    unsigned i;
    for (i = 0; i < n_iter; i++){
        // printf("Iter%d: create task%d\n", i, i);
        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        
        /* We will destroy the task structure by hand so that we can
		 * query the profiling info before the task is destroyed. */
		task->destroy = 0;

        tasks[i] = task;

        ret = starpu_task_submit(task);

        if (STARPU_UNLIKELY(ret == -ENODEV)){
			FPRINTF(stderr, "No worker may execute this task\n");
			exit(0);
		}
    }

    // set synchronization
    starpu_task_wait_for_all();

    // profiling info
    double delay_sum = 0.0;
    double length_sum = 0.0;

    for (i = 0; i < n_iter; i++){
        struct starpu_task *task = tasks[i];
        struct starpu_profiling_task_info *info = task->profiling_info;
        // sum delay time
        delay_sum += starpu_timing_timespec_delay_us(&info->submit_time, &info->start_time);
        // sum exe time
        length_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
        // show worker id
        FPRINTF(stderr, "Task%d : executed on Worker%d\n", i, info->workerid);

        starpu_task_destroy(task);
    }
    free(tasks);

    FPRINTF(stderr, "Avg. delay : %2.2lf us\n", (delay_sum)/n_iter);
	FPRINTF(stderr, "Avg. length : %2.2lf us\n", (length_sum)/n_iter);

    /* Display the occupancy of all workers during the test */
	unsigned worker;
    for (worker = 0; worker < starpu_worker_get_count(); worker++){
        struct starpu_profiling_worker_info worker_info;
        ret = starpu_profiling_worker_get_info(worker, &worker_info);
		STARPU_ASSERT(!ret);

        int n_executed_task = worker_info.executed_tasks;

        char workername[128];
		starpu_worker_get_name(worker, workername, 128);
		FPRINTF(stderr, "Worker %s: executed %d tasks\n", workername, n_executed_task);
    }
    
    return 0;

}