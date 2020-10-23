#include "stencil.h"

/* Heart of the stencil computation: compute a new state from an old one. */
void life_update(int bz, const TYPE *old, TYPE *newp, int nx, int ny, int nz, int ldy, int ldz, int iter)
{
    (void) bz;
    int x, y, z num, alive;
    for (z = iter; z < (nz-iter); z++)
    {
        for (y = K; y < (ny-K); y++)
        {
            for (x = K; x < (nx-K); x++)
            {
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
}