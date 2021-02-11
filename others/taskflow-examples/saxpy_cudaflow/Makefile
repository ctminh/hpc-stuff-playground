CXX_COMPILER=icpc
C_COMPILER=icc
CFLAGS=-std=c++17 -O2 
LDLIBS=-pthread

# for compiling
simple_test: simple_test.o
	$(CXX_COMPILER) $^ $(LDLIBS) -o $@

simple_test.o: simple_test.cpp
	$(CXX_COMPILER) $(CFLAGS) -c $^

run:
	./simple_test

run_tfprof:
	TF_ENABLE_PROFILER=simple_test.json ./simple_test

clean:
	rm -f simple_test *.o