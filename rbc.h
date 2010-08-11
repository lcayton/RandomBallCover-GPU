#ifndef RBC_H
#define RBC_H

#include "defs.h"

void rbc(matrix,matrix,int,int,int*);
void build(const matrix,const matrix,intMatrix,int*,int);
void distSubMat(matrix,matrix,matrix,int,int);

void computeReps(matrix,matrix,int*,real*);
void computeRadii(int*,real*,real*,int,int);
void computeCounts(int*,int,int*);
void computeOffsets(int*,int,int*);
void groupPoints(matrix,int*,int*,int);
void buildQMap(matrix,int*,int*,int,int*);
void blockIntersection(charMatrix,matrix,real*,real*);
void idIntersection(charMatrix);
void fullIntersection(charMatrix);
void initCompPlan(compPlan*,charMatrix,int*,int*,int);
void freeCompPlan(compPlan*);
void computeNNs(matrix,matrix,intMatrix, compPlan,int*,int*,int);




#endif
