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
	include /home/nikola/opt/gasnet/include/mpi-conduit/mpi-par.mak
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

CXXFLAGS = -std=gnu++17 -g -O3 $(BCLFLAGS)
SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	@echo "C $@ $(BACKEND)"
	@time $(CXX) -o $@ $^ $(CXXFLAGS) || echo "$@ $(BACKEND) BUILD FAIL"

run: matrix
	$(BCL_RUN) ./matrix 200 50 4

run_with_inspector: matrix
	$(BCL_RUN) inspxe-cl -c mi2 --result-dir=./ins_logs -- ./matrix 200 50 4

clean:
	rm -f $(TARGETS)
