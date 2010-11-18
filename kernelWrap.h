/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */
#ifndef KERNELWRAP_H
#define KERNELWRAP_H

#include "defs.h"

void dist1Wrap(const matrix,const matrix,matrix);
void kMinsWrap(matrix,matrix,intMatrix);
void findRangeWrap(const matrix,real*,unint);
void rangeSearchWrap(const matrix,const real*,charMatrix);
void nnWrap(const matrix,const matrix,real*,unint*);
void knnWrap(const matrix,const matrix,matrix,intMatrix);
void rangeCountWrap(const matrix,const matrix,real*,unint*);
void planNNWrap(const matrix,const unint*,const matrix,const intMatrix,real*,unint*,compPlan,unint);
void planKNNWrap(const matrix,const unint*,const matrix,const intMatrix,matrix,intMatrix,compPlan,unint);

#endif
