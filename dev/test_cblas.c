/*
 * @Author: shawn233
 * @Date: 2021-03-14 18:40:14
 * @LastEditors: shawn233
 * @LastEditTime: 2021-03-14 19:04:32
 * @Description: Test cblas functions
 */

#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include"./include/cblas.h"


void print_vec(double * vec, int sz) {
    int i = 0;
    for (; i < sz; ++ i) {
        printf("%lf\t", vec[i]);
    }
    printf("\n");
}


int main(void) {

    double A[6] = {1., 2., 3., 4., 5., 6.};
    double B[6] = {7., 8., 9., 10., 11., 12.};
    double C[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
    double D[9] = {1., 1., 1., 1., 1., 1., 1., 1., 1.};

    // cblas_dgemm
    // Conclusion:
    // - when order is CblasRowMajor, lda, ldb and ldc represents the number of **columns**
    //   of the **original** (before transposition) matrx, while M, N and K takes the values
    //   after transposition.
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE trans_A, trans_B;
    blasint M, N, K;
    blasint lda, ldb, ldc;

    // 1 Row Major
    order = CblasRowMajor;
    M = 3;
    N = 3;
    K = 2;

    // 1.1 A*B (No transpose)
    trans_A = CblasNoTrans;
    trans_B = CblasNoTrans;
    lda = 2;
    ldb = 3;
    ldc = 3;
    cblas_dgemm(order, trans_A, trans_B, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    printf("Row major; A No Trans; B No Trans:\n");
    print_vec(C, 9); // row major

    // 1.2 A^T*B (A transpose)
    trans_A = CblasTrans;
    trans_B = CblasNoTrans;
    lda = 3;
    ldb = 3;
    ldc = 3;
    cblas_dgemm(order, trans_A, trans_B, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    printf("Row major; A Trans; B No Trans:\n");
    print_vec(C, 9); // row major

    // 1.3 A*B^T (B transpose)
    trans_A = CblasNoTrans;
    trans_B = CblasTrans;
    lda = 2;
    ldb = 2;
    ldc = 3;
    cblas_dgemm(order, trans_A, trans_B, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    printf("Row major; A No Trans; B Trans:\n");
    print_vec(C, 9); // row major

    // 1.4 A^T*B^T (A, B both transpose)
    trans_A = CblasTrans;
    trans_B = CblasTrans;
    lda = 3;
    ldb = 2;
    ldc = 3;
    cblas_dgemm(order, trans_A, trans_B, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    printf("Row major; A Trans; B Trans:\n");
    print_vec(C, 9); // row major

    // 2 Col Major

    // 2.1 A*B (No transpose)

    // 2.2 A^T*B (A transpose)

    // 2.3 A*B^T (B transpose)

    // 2.4 A^T*B^T (A, B both transpose)
    
    return 0;
}
