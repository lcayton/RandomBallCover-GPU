/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILS_H
#define UTILS_H

#include "defs.h"

void swap(unint*,unint*);
void randPerm(unint,unint*);
void subRandPerm(unint,unint,unint*);
unint randBetween(unint,unint);
void printMat(matrix);
void printMatWithIDs(matrix,unint*);
void printCharMat(charMatrix);
void printIntMat(intMatrix);
void printVector(real*,unint);
void copyVector(real*,real*,unint);
real distVec(matrix,matrix,unint,unint);
double timeDiff(struct timeval,struct timeval);
void copyMat(matrix*,matrix*);
#endif
