MPICC = /home/ctminh/intel/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpicc
CFLAGS = -DTRACE -I$(VT_ROOT)/include -trace
LDLIBS = 

main: main.c
	$(MPICC) $(CFLAGS) main.c -o main $(LDLIBS)

# clean object files and old-itac-traced files
clean:
	rm -fr *.o main main.prot main.stf.*
