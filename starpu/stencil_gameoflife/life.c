#include "stencil.h"

/* Heart of the stencil computation: compute a new state from an old one. */
/*  bz: block z
    old: data_arr
    new: new_data_arr
    nx: num. elements of x-axis
    ny: num. elements of y-axis
    nz: num. elements of z-axis
    ldy: num. elements between two lines
    ldz: num. elements between two plances
 */
void life_update(int bz, const TYPE *old, TYPE *newp, int nx, int ny, int nz, int ldy, int ldz, int iter)
{

    // ------------------------ begin VT -----------------------------
    #ifdef TRACE
    int event_life_update = -1;
    char eventtag_life_update[12] = "life_update";
    int itac_err = VT_funcdef(eventtag_life_update, VT_NOCLASS, &event_life_update);
    VT_BEGIN_CONSTRAINED(event_life_update);
    #endif
    // ---------------------------------------------------------------

    // (void) bz;
    int x, y, z, num, alive;
    for (z = iter; z < (nz-iter); z++)
    {
        for (y = K; y < (ny-K); y++)
        {
            for (x = K; x < (nx-K); x++)
            {
                printf("iter%d: updating block %d -> z = %d, y = %d, x = %d\n", z, bz, z, y, x);

                num = 0
                    + old[x + (y+1)*ldy + (z+0)*ldz]
                    + old[x + (y+1)*ldy + (z+1)*ldz]
                    + old[x + (y+0)*ldy + (z+1)*ldz]
                    + old[x + (y-1)*ldy + (z+1)*ldz]
                    + old[x + (y-1)*ldy + (z+0)*ldz]
                    + old[x + (y-1)*ldy + (z-1)*ldz]
                    + old[x + (y+0)*ldy + (z-1)*ldz]
                    + old[x + (y+1)*ldy + (z-1)*ldz];
                alive = old[x + y*ldy + z*ldz];
                alive = (alive && num == 2) || num == 3;
                newp[x + y*ldy + z*ldz] = alive;
            }
        }
    }

    // ------------------------ end VT -------------------------------
    #ifdef TRACE
    VT_END_W_CONSTRAINED(event_life_update);
    #endif
    // ---------------------------------------------------------------
}