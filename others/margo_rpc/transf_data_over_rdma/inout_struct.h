#define PARAM_H

#include <mercury.h>
#include <mercury_macros.h>

/**
 * The hg_bulk_t opaque type represents a handle to a region of memory in a process.
 * In addition to this handle, we add a field n that will tell us how many
 * values are in the buffer.
 */
MERCURY_GEN_PROC(sum_in_t,
        ((int32_t)(n))\
        ((hg_bulk_t)(bulk)))

MERCURY_GEN_PROC(sum_out_t, ((int32_t)(ret)))

#endif