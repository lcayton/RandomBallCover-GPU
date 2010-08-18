/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "defs.h"
__global__ void planNNKernel(const matrix,const unint*,const matrix,const intMatrix,real*,unint*,compPlan,unint);
__global__ void dist1Kernel(const matrix,unint,const matrix,unint,matrix);
__global__ void nnKernel(const matrix,unint,const matrix,real*,unint*);
__global__ void findRangeKernel(const matrix,unint,real*,unint);
__global__ void rangeSearchKernel(const matrix,unint,unint,const real*,charMatrix);
__global__ void rangeCountKernel(const matrix,unint,const matrix,real*,unint*);

#endif
