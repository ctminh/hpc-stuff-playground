SHELL='bash'

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../

# Error: internal bad pointer BackOff()
# - There could be significant performance degradation on an overloaded system. (OpenMP version)
#
#   This is probably caused by the compiler (g++) failing to optimize the code well in case of system overload. Have to rely on resource allocation from a batch job scheduler (e.g., LSF). Or make sure the cpus are not overloaded if you are running on an interactive machine. 
# 
# - intel compiler 14.0.0 bug: "stl_vector.h(904): internal error: bad pointer emplace_back(_Args&&... __args);"
#
#   This is a bug in intel compiler 14.0.0 (https://software.intel.com/en-us/forums/intel-c-compiler/topic/472385). You can either upgrade your compiler, or use g++ instead. To use `g++`, change `CXX` in `Makefile.inc` to
#
# 	CXX=mpicxx -cxx=g++
#  
# -  error "SEEK_SET is #defined but must not be for the C++ binding of MPI"
#
#   This could happen if your (intel) mpi compiler is not up to date. It is known as a [bug](https://software.intel.com/en-us/articles/intel-cluster-toolkit-for-linux-error-when-compiling-c-aps-using-intel-mpi-library-compilation-driver-mpiicpc) in the MPI-2 standard. Uncomment this line in Makefile.inc to work around it if you cannot upgrade your compiler:
#  
# 	#CXXFLAGS+=-DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX  -idirafter /usr/include/linux #to work around intel compiler bugs

BACKEND = $(shell echo $(BCL_BACKEND) | tr '[:lower:]' '[:upper:]')

TIMER_CMD=time

ifeq ($(BACKEND),SHMEM)
  BACKEND=SHMEM
  BCLFLAGS = -DSHMEM -I$(BCLROOT)
  CXX=oshc++

  BCL_RUN=oshrun -n 4
else ifeq ($(BACKEND),GASNET_EX)
  BACKEND=GASNET_EX
  # XXX: Allow selection of conduit.
  # include /home/nikola/opt/gasnet/include/mpi-conduit/mpi-par.mak
  BCLFLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)
  CXX = mpic++

  BCL_RUN=mpirun -n 4
else ifeq ($(BACKEND), UPCXX)
  BACKEND=UPCXX
  CXX = mpic++

  BCL_RUN=mpirun -n 4
else
  BACKEND=MPI
  BCLFLAGS = -I$(BCLROOT)
  CXX=mpicxx

  BCL_RUN=mpirun -n 4
endif

CXXFLAGS = -std=gnu++17 $(BCLFLAGS)

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) -o $@ $^ $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

test: all
	@for target in $(TARGETS) ; do \
		echo "R $$target $(BACKEND)" ;\
	  time $(BCL_RUN) ./$$target || (echo "$$target $(BACKEND) FAIL $$?"; exit 1) ;\
	done

clean:
	@rm -f $(TARGETS)
