# Compilers
CXX = icpc
CC = icc

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