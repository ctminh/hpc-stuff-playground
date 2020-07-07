# Compilers
CXX = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc
CC = /lrz/sys/intel/studio2019_u5/compilers_and_libraries_2019.5.281/linux/bin/intel64/icc

# Build a static lib
# main_static:	main.o calc.o
# 	$(CC) -o main_static calc.o main.o -lm
# main.o:	main.c
# 	$(CC) -c main.c
# calc.o: calc.c mylib.h
# 	$(CC) -c calc.c

# Build a dynamic lib
main_dynamic:	main.o libcalc.so
	$(CC) -o main_dynamic main.o libcalc.so -lm
main.o: main.c
	$(CC) -c main.c
libcalc.so:	calc.o
	$(CC) -shared -fPIC -o libcalc.so calc.o
calc.o:	calc.c mylib.h
	$(CC) -c calc.c