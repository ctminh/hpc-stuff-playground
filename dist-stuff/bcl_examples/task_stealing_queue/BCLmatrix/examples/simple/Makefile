SHELL='bash'

# XXX: Modify BCLROOT if you move this Makefile
#      out of an examples/* directory.
BCLROOT=$(PWD)/../../
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
  # include /path/to/gasnet/2019.3.0-gcc-8.4.0-j75dwrr/include/mpi-conduit/mpi-par.mak
  BCLFLAGS = $(GASNET_CXXCPPFLAGS) $(GASNET_CXXFLAGS) $(GASNET_LDFLAGS) $(GASNET_LIBS) -DGASNET_EX -I$(BCLROOT)
  CXX = mpicxx -cxx=g++

  BCL_RUN=mpirun -n 4

else ifeq ($(BACKEND), UPCXX)
  BACKEND=UPCXX
  BCLFLAGS=-g -DUPCXX -I$(BCLROOT)
  CXX=upcxx
  
  BCL_RUN=upcxx-run -n 4

else
  BACKEND=MPI
  BCLFLAGS=-I$(BCLROOT)
  CXX=mpiicpc

  BCL_RUN=mpirun -n 4

endif

# c++ flafs for compiling
CXXFLAGS = -std=gnu++17 $(BCLFLAGS)

# declare src and targets
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
