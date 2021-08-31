.PHONY: all

all: fc.c
	make -C ..
	gcc -o fc.out -pthread fc.c ../libopenblas.a -lm
	./fc.out

#test2: test2.c
#	gcc -o test2.out -pthread test2.c ../libopenblas.a -lm

testcblas: test_cblas.c
	gcc -o testcblas.out -pthread test_cblas.c ../libopenblas.a -lm
