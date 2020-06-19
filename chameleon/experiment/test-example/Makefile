# Some declarations
FILE_NAMES ?= multiple-apps.cpp
PROG ?=multiple-apps
TOOL ?= tool

# Compiler
MPICXX = /opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/bin/mpicxx
MPIRUN = /opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/bin/mpirun
CXX = /opt/intel/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Chameleon libs
CH_HOME=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/with_itac/cham_tool

# Parameters for running
OMPI_PARAMETERS = --bind-to-core

# MKL Intel
MKL_INC_DIR = /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/include
MKL_LIB_DIR = /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin
MKL_LINK  = -L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_rt
MKL_INCS  = -I$(MKL_INC_DIR)

# Flags for optimization
APP_COMPILE_FLAGS= -std=c++11
LINKER_FLAGS_CHAM_TOOL= -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL}
LINKER_FLAGS_CHAM_TOOL_CoolMUC = -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL_CoolMUC}

FLAGS_OPTIMIZATION=-g -O3

$(PROG): $(PROG).cpp $(PROG).hpp
	${MPICXX} $(FLAGS_OPTIMIZATION) -o $(PROG) $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(MKL_LINK) $(MKL_INCS)
	
run:
	${MPIRUN} -n 1 ./multiple-apps 20

clean:
	rm -f $(PROG) *.o
