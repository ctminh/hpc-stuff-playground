MPICC = mpicxx
# CFLAGS = -DTRACE -I$(VT_ROOT)/include -trace
LDLIBS = -lhcl

simple_test: simple_test.cpp
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

run:
	mpirun -n 4 ./simple_test

clean:
	rm -fr *.o simple_test *.prot *.stf*
