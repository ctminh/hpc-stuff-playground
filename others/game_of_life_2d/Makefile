MPICC = mpicxx
CFLAGS = -DTRACE -I$(VT_ROOT)/include -trace
LDLIBS = 

life_2D_mpi: life_2D_mpi.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

run:
	mpirun -n 4 ./life_2D_mpi input.txt

# clean object files and old-itac-traced files
clean:
	rm -fr *.o life_2D_mpi *.prot *.stf*
