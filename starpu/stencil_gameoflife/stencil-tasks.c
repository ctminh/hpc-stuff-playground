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
    int iter;
    int bz;
    int niter = get_niter();
    int nbz = get_nbz();
}