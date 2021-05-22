#pragma once

#include <bcl/core/except.hpp>
#include <bcl/core/GlobalPtr.hpp>
#include <bcl/core/malloc.hpp>
#include <bcl/core/alloc.hpp>

#ifdef SHMEM
  #include <bcl/backends/shmem/backend.hpp>
#elif GASNET_EX
  #include <bcl/backends/gasnet-ex/backend.hpp>
#elif UPCXX
  #include <bcl/backends/upcxx/backend.hpp>
#else
  #include <bcl/backends/mpi/backend.hpp>
#endif

#define MSIZE 150

#include <bcl/core/comm.hpp>
#include <bcl/core/teams.hpp>
#include <bcl/core/util.hpp>


struct task
{
    int matrixSize;
    double matrix[MSIZE*MSIZE];
    double matrix2[MSIZE*MSIZE];
    double result[MSIZE*MSIZE];
    unsigned int taskId;
};

namespace BCL {
  // TODO: put these in a compilation unit.
  uint64_t shared_segment_size;
  void *smem_base_ptr;
}
