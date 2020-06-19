# Some declarations
FILE_NAMES ?= mxm_unequal_tasks.cpp
PROG ?=mxm_unequal_tasks
TOOL ?= tool

# Compiler
MPICXX = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/lrzbin/mpicxx
MPIRUN = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/bin/mpirun
CXX = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc
VT_ROOT = /lrz/sys/intel/studio2019_u5/itac/2019.5.041

# Compiler on CoolMUC
MPICXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpicxx
MPIRUN_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpirun
CXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel/19.0.5-gcc-fcpqmql/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Chameleon libs
CH_HOME=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/with_itac/cham_tool
CH_LIB_TOOL=$(CH_HOME)/lib
CH_HOME_CoolMUC = /dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/cham_tool
CH_LIB_TOOL_CoolMUC = $(CH_HOME_CoolMUC)/lib

# Parameters for running
OMPI_PARAMETERS = --bind-to-core 

# Flags for optimization
APP_COMPILE_FLAGS= -std=c++11
LINKER_FLAGS_CHAM_TOOL= -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL}
LINKER_FLAGS_CHAM_TOOL_CoolMUC = -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL_CoolMUC}

FLAGS_OPTIMIZATION=-g -O3

$(PROG)_tool: $(PROG).cpp
	${MPICXX} $(FLAGS_OPTIMIZATION) -o $(PROG)_tool $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL)

$(PROG)_tool_coolmuc: $(PROG).cpp
	${MPICXX_CoolMUC} $(FLAGS_OPTIMIZATION) -o $(PROG)_tool_coolmuc $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL_CoolMUC)

$(TOOL): $(TOOL).cpp $(TOOL).h
	$(MPICXX) -g -O0 -shared -std=c++11 -fPIC -o $(TOOL).so $(TOOL).cpp -L$(CH_HOME)/lib -I$(CH_HOME)include -I$(VT_ROOT)/include -trace

$(TOOL)_coolmuc: $(TOOL).cpp $(TOOL).h
	${MPICXX_CoolMUC} -g -O0 -shared -std=c++11 -fPIC -o $(TOOL)_coolmuc.so $(TOOL).cpp -L$(CH_HOME_CoolMUC)/lib -I$(CH_HOME_CoolMUC)include

clean:
	rm -f $(PROG)_* $(TOOL).so *.o
