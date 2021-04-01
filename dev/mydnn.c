/*
 * @Author: shawn233
 * @Date: 2021-03-12 00:12:01
 * @LastEditors: shawn233
 * @LastEditTime: 2021-03-12 00:12:27
 * @Description: file content
 */


#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include"../source/cblas.h"


struct FC_layer {
    // (parameters) W: weight, b: bias, sigma: activation function
    double *W, *b;
    double * (* sigma)(double *);
    // (intermediate results) Z = XW + B, A = sigma(Z)
    double *Z, *A; 
};

typedef struct FC_layer FC_layer;
