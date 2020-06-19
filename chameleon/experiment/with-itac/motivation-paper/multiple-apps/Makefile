# Some declarations
FILE_NAMES ?= multiple-apps.cpp
PROG ?=multiple-apps
TOOL ?= tool

# Compiler
MPICXX = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/lrzbin/mpicxx
MPIRUN = /lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/bin/mpirun
CXX = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Compiler on CoolMUC
# MPICXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpicxx
# MPIRUN_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel-mpi/2019.6.166-gcc-f5wk3ki/compilers_and_libraries_2020.0.166/linux/mpi/intel64/bin/mpirun
# CXX_CoolMUC = /dss/dsshome1/lrz/sys/spack/staging/20.1/opt/x86_64/intel/19.0.5-gcc-fcpqmql/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc

# Chameleon libs
CH_HOME=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/install/with_itac/cham_tool
CH_HOME_CoolMUC = /dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/cham_tool
CH_LIB_TOOL = $(CH_HOME)/lib
CH_LIB_TOOL_CoolMUC = $(CH_HOME_CoolMUC)/lib


# Parameters for running
OMPI_PARAMETERS = --bind-to-core

# MKL Intel
MKL_INC_DIR = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/mkl/include
MKL_LIB_DIR = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64
MKL_LINK  = -L$(MKL_LIB_DIR) -lmkl_sequential -lmkl_core -lmkl_rt
MKL_INCS  = -I$(MKL_INC_DIR)

# Flags for optimization
APP_COMPILE_FLAGS= -std=c++11
LINKER_FLAGS_CHAM_TOOL= -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL}
LINKER_FLAGS_CHAM_TOOL_CoolMUC = -fopenmp -lm -lstdc++ -lchameleon -L${CH_LIB_TOOL_CoolMUC}

FLAGS_OPTIMIZATION=-g -O3

$(PROG): $(PROG).cpp $(PROG).hpp
	${MPICXX} $(FLAGS_OPTIMIZATION) -o $(PROG) $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL) $(MKL_LINK) $(MKL_INCS)

# $(PROG)_coolmuc: $(PROG).cpp
# 	${MPICXX_CoolMUC} $(FLAGS_OPTIMIZATION) -o $(PROG)_coolmuc $(APP_COMPILE_FLAGS) $(FILE_NAMES) $(LINKER_FLAGS_CHAM_TOOL_CoolMUC)


clean:
	rm -f $(PROG) $(PROG)_coolmuc *.o
