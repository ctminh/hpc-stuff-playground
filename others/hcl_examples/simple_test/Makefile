MPICC = mpicxx
LDLIBS = -lhcl -L/dss/dsshome1/0A/di49mew/hcl/install/lib64

simple_test: simple_test.cpp util.h
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

run:
	mpirun -n 4 ./simple_test

clean:
	rm -fr *.o simple_test *.prot *.stf*
