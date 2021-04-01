.PHONY: all

all: fc.c
	gcc -o fc -pthread fc.c ../libopenblas.a -lm

#test2: test2.c
#	gcc -o test2 -pthread test2.c ../libopenblas.a -lm

testcblas: test_cblas.c
	gcc -o testcblas -pthread test_cblas.c ../libopenblas.a -lm
