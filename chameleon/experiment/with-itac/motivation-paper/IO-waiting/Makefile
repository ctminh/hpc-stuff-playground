# Some declarations
FILE_NAMES ?= exp_io_waiting.cpp
PROG ?=exp_io_waiting
TOOL ?= tool
NUM_TASKS_R0 ?= 100	# for make run
NUM_TASKS_R1 ?= 100 # for make run
NUM_RANKS ?= 2

# Compiler
MPICXX = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/lrzbin/mpicxx
MPIRUN = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/bin/mpirun
CXX = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Compiler on CoolMUC
MPICXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpicxx
MPIRUN_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpirun
CXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel/19.0.5-gcc-fcpqmql/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Chameleon libs
CH_HOME=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/with_itac/cham_tool
CH_HOME_CoolMUC = /dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/cham_tool
CH_LIB_TOOL = $(CH_HOME)/lib
CH_LIB_TOOL_CoolMUC = $(CH_HOME_CoolMUC)/lib


# Parameters for running
OMPI_PARAMETERS = --bind-to-core 

# Flags for optimization
APP_COMPILE_FLAGS= -std=c++11
LINKER_FLAGS_CHAM_TOOL= -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL}
LINKER_FLAGS_CHAM_TOOL_CoolMUC = -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL_CoolMUC}

FLAGS_OPTIMIZATION=-g -O3

$(PROG): $(PROG).cpp
	${MPICXX} $(FLAGS_OPTIMIZATION) -o $(PROG) $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL)

$(PROG)_coolmuc: $(PROG).cpp
	${MPICXX_CoolMUC} $(FLAGS_OPTIMIZATION) -o $(PROG)_coolmuc $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL_CoolMUC)


clean:
	rm -f $(PROG) $(PROG)_coolmuc *.o
