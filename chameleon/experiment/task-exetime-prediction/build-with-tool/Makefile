# Some declarations
TOOL ?= tool

# Compiler on CoolMUC
MPICXX_CoolMUC = /lrz/sys/tools/intel-mpi-wrappers/bin/mpicxx
MPIRUN_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1.1/opt/x86_64/intel-mpi/2019.7.217-gcc-vf2p2hg/compilers_and_libraries_2020.1.217/linux/mpi/intel64/bin/mpirun
CXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1.1/opt/x86_64/intel/19.0.5-gcc-fcpqmql/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Chameleon libs
CH_HOME_CoolMUC = /dss/dsshome1/lxc0D/ra56kop/task-based-models-survey/chameleon/install/cham_tool_itac
CH_LIB_TOOL_CoolMUC = $(CH_HOME_CoolMUC)/lib

# Parameters for running
# OMPI_PARAMETERS = --bind-to-core 

# Flags for optimization
APP_COMPILE_FLAGS= -std=c++11
LINKER_FLAGS_CHAM_TOOL= -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL}
LINKER_FLAGS_CHAM_TOOL_CoolMUC = -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL_CoolMUC}

FLAGS_OPTIMIZATION=-g -O3

# $(MPICXX) -g -O0 -shared -std=c++11 -fPIC -o $(TOOL).so $(TOOL).cpp -L$(CH_HOME)/lib -I$(CH_HOME)include -I$(VT_ROOT)/include -trace

$(TOOL)_coolmuc: $(TOOL).cpp $(TOOL).h
	${MPICXX_CoolMUC} -g -O0 -shared -std=c++11 -fPIC -o $(TOOL)_coolmuc.so $(TOOL).cpp -L$(CH_HOME_CoolMUC)/lib -I$(CH_HOME_CoolMUC)include

clean:
	rm -f $(TOOL).so *.o
