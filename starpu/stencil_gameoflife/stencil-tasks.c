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


/*
 * Create all the tasks
 */
void create_tasks(int rank)
{
    #ifdef TRACE
    int event_taskcreate = -1;
    char event_createtask[12] = "create_task";
    int itac_err = VT_funcdef(event_createtask, VT_NOCLASS, &event_taskcreate);
    #endif

    int iter;
    int bz;
    int niter = get_niter();
    int nbz = get_nbz();
}