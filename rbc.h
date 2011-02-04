/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef RBC_H
#define RBC_H

#include "defs.h"


void buildRBC(const matrix,rbcStruct*,unint, unint);
void queryRBC(const matrix,const rbcStruct,unint*,real*);
void kqueryRBC(const matrix,const rbcStruct,intMatrix,matrix);
void destroyRBC(rbcStruct*);
void distSubMat(matrix,matrix,matrix,unint,unint);
void computeReps(matrix,matrix,unint*,real*);
void computeRadii(unint*,real*,real*,unint,unint);
void computeCounts(unint*,unint,unint*);
void buildQMap(matrix,unint*,unint*,unint,unint*);
void idIntersection(charMatrix);
void fullIntersection(charMatrix);
void initCompPlan(compPlan*,charMatrix,unint*,unint*,unint);
void freeCompPlan(compPlan*);
void computeNNs(matrix,intMatrix,matrix,unint*,compPlan,unint*,real*,unint);
void computeKNNs(matrix,intMatrix,matrix,unint*,compPlan,intMatrix,matrix,unint);
void setupReps(matrix,rbcStruct*,int);

#endif
