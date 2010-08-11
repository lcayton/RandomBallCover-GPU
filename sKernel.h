#ifndef SKERNEL_H
#define SKERNEL_H

#include "defs.h"
__global__ void sumKernelI(intMatrix,intMatrix,intMatrix,int);
__global__ void sumKernel(charMatrix,intMatrix,intMatrix,int);
__global__ void combineSumKernel(intMatrix,intMatrix,int);
__global__ void getCountsKernel(int*,charMatrix,intMatrix);
void getCountsWrap(int*,charMatrix,intMatrix);
void buildMapWrap(intMatrix,charMatrix,intMatrix,int);
__global__ void buildMapKernel(intMatrix,charMatrix,intMatrix,int);
void sumWrap(charMatrix,intMatrix);

#endif
