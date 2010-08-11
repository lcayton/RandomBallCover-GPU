#ifndef KERNELWRAP_H
#define KERNELWRAP_H

#include "defs.h"

void dist1Wrap(matrix,matrix,matrix);
void kMinsWrap(matrix,matrix,intMatrix);
void findRangeWrap(matrix,real*,int);
void rangeSearchWrap(matrix,real*,charMatrix);
void nnWrap(const matrix,const matrix,real*,int*);
void rangeCountWrap(const matrix,const matrix,real*,int*);
void pruneWrap(charMatrix,matrix,real*,real*);
#endif
