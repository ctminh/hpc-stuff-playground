#include <starpu.h>
#include <starpu_scheduler.h>

#define NTASKS 32


////////////////////////////////////////////////////////////
/* task function */
void dummy_func(void *descr[] STARPU_ATTRIBUTE_UNUSED, void *arg STARPU_ATTRIBUTE_UNUSED)
{
    
}

////////////////////////////////////////////////////////////
/* codelet */
static struct starpu_codelet dummy_codelet = 
{
    .cpu_funcs = {dummy_func},
    .cpu_funcs_name = {"dummy_func"},
    .model = NULL,
    .nbuffers = 0,
    .name = "dummy",
};

////////////////////////////////////////////////////////////
/* main function */
int main(int argc, char *argv[])
{
    int ntasks = NTASKS;
    int ret;
    struct starpu_conf conf;

    int i;
    for (i = 0; i < ntasks; i++)
    {
        struct starpu_task *task = starpu_task_create();
        task->cl = &dummy_codelet;
        task->cl_arg = NULL;
        ret = starpu_task_submit(task);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    }

    return 0;
}