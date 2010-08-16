#ifndef UTILS_H
#define UTILS_H

#include "defs.h"

void swap(int*,int*);
void randPerm(int,int*);
void subRandPerm(int,int,int*);
int randBetween(int,int);
void printMat(matrix);
void printMatWithIDs(matrix,int*);
void printCharMat(charMatrix);
void printIntMat(intMatrix);
void printVector(real*,int);
void copyVector(real*,real*,int);
real distL1(matrix,matrix,int,int);
double timeDiff(struct timeval,struct timeval);
void copyMat(matrix*,matrix*);
#endif
