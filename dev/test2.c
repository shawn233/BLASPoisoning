/*
 * @Author: shawn233
 * @Date: 2021-03-14 17:34:59
 * @LastEditors: shawn233
 * @LastEditTime: 2021-03-14 18:36:20
 * @Description: file content
 */
#include"../source/cblas.h"
#include<stdio.h>
#include<stdlib.h>

void main()
{
	int i = 0;
	double A[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
	double B[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
	double C[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
	double D[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
	double E[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
	double F[9] = { 0., 0., 0., 0., 0., 0., 0., 0., 0. };
	
	/*按列主序展开*/
	//1、都无转置
	// printf("按列主序展开,都无转置：\n");
	// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1, A, 3, B, 2, 1, C, 3);
	// for (i = 0; i<9; i++)
	// 	printf("%lf ", C[i]);
	// printf("\n");
 
	// //2\矩阵B转置
	// printf("按列主序展开，矩阵B转置：\n");
	// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3, 1, D, 3);
	// for (i = 0; i<9; i++)
	// 	printf("%lf ", D[i]);
	// printf("\n");
 
	/*按行主序展开*/
	//1、都无转置
	printf("按行主序展开,都无转置:\n");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 2, 1, A, 2, B, 3, 1, E, 3);
	for (i = 0; i<9; i++)
		printf("%lf ",E[i]);
	printf("\n");
 
	//2、矩阵B转置
	printf("按行主序展开,矩阵B转置:\n");
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 2, B, 2, 1, F, 3);
	for (i = 0; i<9; i++)
		printf("%lf ",F[i]);
	printf("\n");
}

