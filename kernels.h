#ifndef KERNELS_H
#define KERNELS_H

#include "defs.h"
__global__ void planNNKernel(const matrix,const matrix,real*,int*,compPlan,intMatrix,int*,int);
__global__ void pruneKernel(const matrix,const real*,const real*,charMatrix);
__global__ void dist1Kernel(const matrix,const matrix,matrix);
__device__ matrix getSubMat(matrix,int,int);
__global__ void nnKernel(const matrix,const matrix,real*,int*);

__global__ void findRangeKernel(matrix,real*,int);
__global__ void rangeSearchKernel(matrix,real*,charMatrix);
__global__ void rangeCountKernel(const matrix,const matrix,real*,int*);

#endif
