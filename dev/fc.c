/*
 * @Author: shawn233
 * @Date: 2021-03-05 19:11:15
 * @LastEditors: shawn233
 * @LastEditTime: 2021-04-09 20:27:38
 * @Description: Logistic Regression in OpenBLAS
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "./include/cblas.h"

#define ALLOC(type, var) net.var = (type *)malloc(net.size_##var * sizeof(type))
#define ALLOC_DOUBLE(var) net.var = (double *)malloc(net.size_##var * sizeof(double))
#define DEBUG false


/* 
 * NOTE: X = [x1, x2, ..., xN]^T
 * one sample per **row**
 */

struct hyperparams {
    int n_features;
    int n_labels;
    int n_samples;
    int batch_size;
    int n_epochs;
    double lr;
};

typedef struct hyperparams hyperparams;


struct network {
    // A: weight, bias: bias, Z: activation, Y: outputs
    double *A, *bias, *Z, *Y;
    size_t size_A, size_bias, size_Z, size_Y;
    
    // exp_Z: exp(Z)
    // row_sum_exp_Z: sum of exponential in softmax
    // softmax_sum_op: [1 1 .. 1] as the summation vector in softmax
    double *exp_Z, *rowsum_exp_Z, *softmax_sum_op;
    size_t size_exp_Z, size_rowsum_exp_Z, size_softmax_sum_op;

    // for backward propogation
    double *delta, *A_update, *bias_update;
    size_t size_delta, size_A_update, size_bias_update;
};

typedef struct network network;


struct dataset {
    double *features;
    int *labels;
    size_t size_features, size_labels;
};

typedef struct dataset dataset;


void destroy(network *net, dataset *data) {
    free(net->A);
    free(net->bias);
    free(net->Z);
    free(net->Y);

    free(net->exp_Z);
    free(net->rowsum_exp_Z);
    free(net->softmax_sum_op);
    
    free(net->A_update);
    free(net->delta);
    free(net->bias_update);
    
    free(data->features);
    free(data->labels);
}


void print_vec_as_matrix(const char *name, const double *vec, const int nrows, const int ncols) {
    int i, j;
    printf("\n%s\n", name);
    for (i = 0; i < nrows; ++ i) {
        for (j = 0; j < ncols; ++ j)
            printf("%.4lf\t", vec[ncols*i+j]);
        printf("\n");
    }
}


void print_vec_as_matrix_int(const char *name, const int *vec, const int nrows, const int ncols) {
    int i, j;
    printf("\n%s\n", name);
    for (i = 0; i < nrows; ++ i) {
        for (j = 0; j < ncols; ++ j)
            printf("%d\t", vec[ncols*i+j]);
        printf("\n");
    }
}


void elementwise_exponential(double *vec, double *res, const size_t sz) {
    int i = 0;
    for (; i < sz; ++ i)
        res[i] = exp(vec[i]);
}


void softmax(network *net, const size_t batch_size, const size_t n_labels) {
    int i, j;

    // exp_Z = exp(Z)
    elementwise_exponential(net->Z, net->exp_Z, net->size_Z);
    if (DEBUG)
        print_vec_as_matrix("MATRIX exp_Z", net->exp_Z, batch_size, n_labels);
    
    // rowsum_exp_Z = exp_Z * [1 1 .. 1]^T
    blasint lda = n_labels;
    cblas_dgemv( \
        CblasRowMajor, CblasNoTrans, batch_size, n_labels, 1.0, net->exp_Z, \
        lda, net->softmax_sum_op, 1, 0.0, net->rowsum_exp_Z, 1);
    
    if (DEBUG)
        print_vec_as_matrix("MATRIX rowsum_exp_Z", net->rowsum_exp_Z, batch_size, 1);

    // Y[i, j] = exp_Z[i, j] / rowsum_exp_Z[i]
    for (i = 0; i < batch_size; ++ i) {
        for (j = 0; j < n_labels; ++ j) {
            net->Y[i * n_labels + j] = net->exp_Z[i * n_labels + j] / net->rowsum_exp_Z[i];
        }
    }
}


void forward(network *net, hyperparams *hp, double *X_batch) {
    int i, j;

    // copy bias to fill matric Z 
    for (i = 0; i < hp->batch_size; ++ i)
        cblas_dcopy(net->size_bias, net->bias, 1, net->Z + i * net->size_bias, 1);

    // print_vec_as_matrix("MATRIX X_batch", X_batch, hp->batch_size, hp->n_features);
    // print_vec_as_matrix("MATRIX B", net->Z, hp->batch_size, hp->n_labels);

    blasint lda = hp->n_features;
    blasint ldb = hp->n_features;
    blasint ldc = hp->n_labels; 

    // Z = X * A + Z
    cblas_dgemm( \
        CblasRowMajor, CblasNoTrans, CblasTrans, hp->batch_size, hp->n_labels, hp->n_features, \
        1.0, X_batch, lda, net->A, ldb, 1.0, net->Z, ldc);
    
    if (DEBUG)
        print_vec_as_matrix("MATRIX Z", net->Z, hp->batch_size, hp->n_labels);

    // Y = softmax(Z)
    softmax(net, hp->batch_size, hp->n_labels);

    if (DEBUG)
        print_vec_as_matrix("MATRIX Y", net->Y, hp->batch_size, hp->n_labels);
}


// double backward(network *net, hyperparams *hp, int *batch_labels, double *X_batch) {
//     /* label \in [0, hp->n_labels - 1] */
//     int i, j, idx;
//     int label;
//     double log_yc, partial_az, rowsum_exp_Z;
//     double celoss = 0.0;

//     // delta = - partial_az / logY[label]
//     for (i = 0; i < hp->batch_size; ++ i) {
//         label = batch_labels[i];
//         rowsum_exp_Z = net->rowsum_exp_Z[i];
//         log_yc = log(net->Y[i * hp->n_labels + label]);
//         celoss += - log_yc;
        
//         for (j = 0; j < hp->n_labels; ++ j) {
//             idx = i * hp->n_labels + j;
//             if (j == label) {
//                 partial_az = net->exp_Z[idx] * rowsum_exp_Z - pow(net->exp_Z[idx], 2.0);
//                 partial_az = partial_az / pow(rowsum_exp_Z, 2.0);
//             } else {
//                 partial_az = - net->exp_Z[i * hp->n_labels + label] * net->exp_Z[idx];
//                 partial_az = partial_az / pow(rowsum_exp_Z, 2.0);
//             }
            
//             net->delta[idx] = - partial_az / log_yc;
//         }
        
//     }

//     celoss /= hp->batch_size;

//     if (DEBUG)
//         print_vec_as_matrix("MATRIX delta", net->delta, hp->batch_size, hp->n_labels);

//     // A_update = X^T * delta / batch_size
//     blasint lda = hp->n_features;
//     blasint ldb = hp->n_labels;
//     blasint ldc = hp->n_labels;

//     cblas_dgemm( \
//         CblasRowMajor, CblasTrans, CblasNoTrans, hp->n_features, hp->n_labels, hp->batch_size, \
//         1.0, X_batch, lda, net->delta, ldb, 0.0, net->A_update, ldc);

//     for (i = 0; i < net->size_A_update; ++ i) {
//         net->A_update[i] /= (double)hp->batch_size;
//     }

//     if (DEBUG)
//         print_vec_as_matrix("MATRIX A_update", net->A_update, hp->n_features, hp->n_labels);

//     // bias_update = avg(delta, dim=0)
//     for (i = 0; i < net->size_bias_update; ++ i) {
//         net->bias_update[i] = 0.0;
//     }
//     for (i = 0; i < hp->batch_size; ++ i) {
//         for (j = 0; j < hp->n_labels; ++ j) {
//             net->bias_update[j] += net->delta[i * hp->n_labels + j];
//         }
//     }
//     for (i = 0; i < net->size_bias_update; ++ i) {
//         net->bias_update[i] /= (double)hp->batch_size;
//     }

//     if (DEBUG) {
//         print_vec_as_matrix("MATRIX bias_update", net->bias_update, 1, hp->n_labels);
//     }
    
//     cblas_daxpy(net->size_A, -hp->lr, net->A_update, 1, net->A, 1);
//     cblas_daxpy(net->size_bias, -hp->lr, net->bias_update, 1, net->bias, 1);
//     if (hp->lr > 0.01)
//         hp->lr = hp->lr * 0.5;

//     if (DEBUG) {
//         print_vec_as_matrix("MATRIX A (update)", net->A, hp->n_features, hp->n_labels);
//         print_vec_as_matrix("MATRIX bias (update)", net->bias, 1, hp->n_labels);
//     }

//     return celoss;
// }


double loss(hyperparams *hp, network *net, int *batch_Y) {
    
}


double accuracy(hyperparams *hp, network *net, int *batch_Y) {
    int i, j, pred;
    int total_correct = 0;

    for (i = 0; i < hp->batch_size; ++ i) {
        pred = 0;
        for (j = 0; j < hp->n_labels; ++ j) {
            if (net->Y[i * hp->n_labels + j] > net->Y[i * hp->n_labels + pred])
                pred = j;
        }
        if (pred == batch_Y[i]) {
            ++ total_correct;
        }
    }

    return (double)total_correct / hp->batch_size;
}


void test(hyperparams *hp, network *net, dataset *data) {
    int n_batch;
    double *batch_X;
    int *batch_Y;
    double acc;
    double running_loss, running_acc;

    int n_batches = hp->n_samples / hp->batch_size;
    
    running_loss = running_acc = 0.0;
    
    for (n_batch = 0; n_batch < n_batches; ++ n_batch) {
        batch_X = data->features + n_batch * hp->batch_size * hp->n_features;
        batch_Y = data->labels + n_batch * hp->batch_size;

        if (DEBUG) {
            print_vec_as_matrix("MATRIX BATCH_X", batch_X, hp->batch_size, hp->n_features);
            print_vec_as_matrix_int("MATRIX BATCH_Y", batch_Y, hp->batch_size, 1);
        }
        
        forward(net, hp, batch_X);
        // celoss = backward(net, hp, batch_Y, batch_X);
        acc = accuracy(hp, net, batch_Y);
        
        // running_loss += celoss;
        running_acc += acc;
    }

    printf(
        "test loss: %.4lf acc: %.4lf\n", 
        running_loss / n_batches, running_acc / n_batches);
}


// void train(hyperparams *hp, network *net, dataset *data) {
//     int epoch, n_batch;
//     double *batch_X;
//     int *batch_Y;
//     double celoss, acc;
//     double running_loss, running_acc;

//     int n_batches = hp->n_samples / hp->batch_size;
    
//     for (epoch = 0; epoch < hp->n_epochs; ++ epoch) {
//         running_loss = running_acc = 0.0;
        
//         for (n_batch = 0; n_batch < n_batches; ++ n_batch) {
//             batch_X = data->features + n_batch * hp->batch_size * hp->n_features;
//             batch_Y = data->labels + n_batch * hp->batch_size;

//             if (DEBUG) {
//                 print_vec_as_matrix("MATRIX BATCH_X", batch_X, hp->batch_size, hp->n_features);
//                 print_vec_as_matrix_int("MATRIX BATCH_Y", batch_Y, hp->batch_size, 1);
//             }
            
//             forward(net, hp, batch_X);
//             celoss = backward(net, hp, batch_Y, batch_X);
//             acc = accuracy(hp, net, batch_Y);
            
//             running_loss += celoss;
//             running_acc += acc;
//         }

//         printf(
//             "epoch %d / %d loss: %.4lf acc: %.4lf\n", 
//             epoch, hp->n_epochs, running_loss / n_batches, running_acc / n_batches);
//     } 
// }


int main(int argc, char *argv[]) {
    const int MAX_PRINT_SAMPELS = 20;
    if (DEBUG) {
        printf("\n%d argument(s)\n", argc);
    }
    
    // hyperparameters
    hyperparams hp;

    hp.n_features = 4;
    hp.n_labels = 3;
    hp.n_samples = 150;
    hp.batch_size = 75;
    hp.n_epochs = 100;
    hp.lr = 0.1;

    // parameters
    network net;

    net.size_A = hp.n_features * hp.n_labels;
    net.size_bias = hp.n_labels;
    net.size_Z = hp.batch_size * hp.n_labels;
    net.size_Y = hp.batch_size * hp.n_labels;

    net.size_exp_Z = net.size_Z;
    net.size_rowsum_exp_Z = hp.batch_size;
    net.size_softmax_sum_op = hp.n_labels;
    
    net.size_delta = hp.batch_size * net.size_bias;
    net.size_A_update = net.size_A;
    net.size_bias_update = net.size_bias;

    ALLOC_DOUBLE(A);
    ALLOC_DOUBLE(bias);
    ALLOC_DOUBLE(Z);
    ALLOC_DOUBLE(Y);

    ALLOC_DOUBLE(exp_Z);
    ALLOC_DOUBLE(rowsum_exp_Z);
    ALLOC_DOUBLE(softmax_sum_op);
    
    ALLOC_DOUBLE(delta);
    ALLOC_DOUBLE(A_update);
    ALLOC_DOUBLE(bias_update);


    if (DEBUG) {
        printf("\nAllocation\tA\tbias\tZ\n");
        printf( \
            "Matrix size 2d\t%dx%d\t%d\t%dx%d\n", \
            hp.n_features, hp.n_labels, hp.n_labels, \
            hp.batch_size, hp.n_labels);
        printf( \
            "Matrix size 1d\t%lu\t%lu\t%lu\n", \
            net.size_A, net.size_bias, net.size_Z);
    }

    // initialization
    int i = 0;
    for (i = 0; i < net.size_A; ++ i) {
        net.A[i] = 0.5;
    }
    for (i = 0; i < net.size_bias; ++ i) {
        net.bias[i] = 0.5;
    }
    // print_vec_as_matrix("MATRIX bias", net.bias, 1, hp.n_labels);

    /* no need to initialize Z, X and Y */
    for (i = 0; i < net.size_softmax_sum_op; ++ i) {
        net.softmax_sum_op[i] = 1.0;
    }
    
    // load dataset
    FILE *fp_features, *fp_labels;
    dataset iris; 
    
    iris.size_features = hp.n_samples * hp.n_features;
    iris.size_labels = hp.n_samples;
    iris.features = (double *)malloc(iris.size_features * sizeof(double));
    iris.labels = (int *)malloc(iris.size_labels * sizeof(int));

    fp_features = fopen("./data/iris/iris-features.txt", "r");
    fp_labels = fopen("./data/iris/iris-labels.txt", "r");

    // fp_features = fopen("./data/gen/gen-features.txt", "r");
    // fp_labels = fopen("./data/gen/gen-labels.txt", "r");
    
    /* Can we automatically produce the format string? */
    int j;
    for (i = 0; i < hp.n_samples; ++ i) {
        for (j = 0; j < hp.n_features; ++ j) {
            fscanf(fp_features, "%lf", &iris.features[hp.n_features * i + j]);
            fgetc(fp_features);
        }
    }

    for (i = 0; i < hp.n_samples; ++ i) {
        fscanf(fp_labels, "%d\n", &iris.labels[i]);
    }

    fclose(fp_features);
    fclose(fp_labels);

    // print out loaded data
    // if (DEBUG) {
    //     int max_print_features = (hp.n_samples > MAX_PRINT_SAMPELS)? MAX_PRINT_SAMPELS: hp.n_samples;
    //     print_vec_as_matrix("FEATURES", iris.features, max_print_features, hp.n_features);
    //     print_vec_as_matrix_int("LABELS", iris.labels, max_print_features, 1);
    // }

    // Load external parameters
    FILE *fp_params;
    fp_params = fopen("./model/iris/best.txt", "r");

    fscanf(fp_params, "WEIGHT:\n");
    for (i = 0; i < hp.n_labels; ++ i) {
        for (j = 0; j < hp.n_features; ++ j) {
            fscanf(fp_params, "%lf", &net.A[i * hp.n_features + j]);
        }
        fscanf(fp_params, "\n");
    }

    fscanf(fp_params, "BIAS:\n");
    for (i = 0; i < hp.n_labels; ++ i) {
        fscanf(fp_params, "%lf", &net.bias[i]);
    }
    fscanf(fp_params, "\n");

    fclose(fp_params);

    print_vec_as_matrix("A LOAD", net.A, hp.n_labels, hp.n_features);
    print_vec_as_matrix("bias LOAD", net.bias, 1, hp.n_labels);

    test(&hp, &net, &iris);


    printf("\nDone!\n");
    destroy(&net, &iris);
    
    return 0;
}


