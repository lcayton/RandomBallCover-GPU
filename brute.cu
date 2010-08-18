/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_CU
#define BRUTE_CU

#include "utilsGPU.h"
#include "utils.h"
#include "rbc.h"
#include "defs.h"
#include "kernels.h"
#include "kernelWrap.h"
#include "brute.h"
#include<stdio.h>
#include<cuda.h>

void bruteRangeCount(matrix x, matrix q, real *ranges, unint *cnts){
  matrix dx, dq;
  real *dranges;
  unint *dcnts;
  
  copyAndMove(&dx, &x);
  copyAndMove(&dq, &q);

  checkErr( cudaMalloc( (void**)&dranges, q.pr*sizeof(*dranges) ) );
  cudaMemcpy( dranges, ranges, q.r*sizeof(*dranges), cudaMemcpyHostToDevice );

  checkErr( cudaMalloc( (void**)&dcnts, q.pr*sizeof(*dcnts) ) );
  
  rangeCountWrap(dq, dx, dranges, dcnts);
  
  cudaMemcpy(cnts, dcnts, q.r*sizeof(*cnts), cudaMemcpyDeviceToHost );

  cudaFree(dcnts);
  cudaFree(dranges);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
}


void bruteSearch(matrix x, matrix q, unint *NNs){
  real *dMins;
  unint *dMinIDs;
  matrix dx, dq;

  
  dx.r=x.r; dx.pr=x.pr; dx.c=x.c; dx.pc=x.pc; dx.ld=x.ld;
  dq.r=q.r; dq.pr=q.pr; dq.c=q.c; dq.pc=q.pc; dq.ld=q.ld;

  checkErr( cudaMalloc((void**)&dMins, q.pr*sizeof(*dMins)) );
  checkErr( cudaMalloc((void**)&dMinIDs, q.pr*sizeof(*dMinIDs)) );
  checkErr( cudaMalloc((void**)&dx.mat, dx.pr*dx.pc*sizeof(*dx.mat)) );
  checkErr( cudaMalloc((void**)&dq.mat, dq.pr*dq.pc*sizeof(*dq.mat)) );

  cudaMemcpy(dx.mat,x.mat,x.pr*x.pc*sizeof(*dx.mat),cudaMemcpyHostToDevice);
  cudaMemcpy(dq.mat,q.mat,q.pr*q.pc*sizeof(*dq.mat),cudaMemcpyHostToDevice);
  
  nnWrap(dq,dx,dMins,dMinIDs);

  cudaMemcpy(NNs,dMinIDs,dq.r*sizeof(*NNs),cudaMemcpyDeviceToHost);
  
  cudaFree(dMins);
  cudaFree(dMinIDs);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
}


void bruteCPU(matrix X, matrix Q, unint *NNs){
  real *dtoNNs; 
  real temp;

  unint i, j;

  dtoNNs = (real*)calloc(Q.r,sizeof(*dtoNNs));
  
  for( i=0; i<Q.r; i++ ){
    dtoNNs[i] = MAX_REAL;
    NNs[i] = 0;
    for(j=0; j<X.r; j++ ){
      temp = distVec( Q, X, i, j );
      if( temp < dtoNNs[i]){
	NNs[i] = j;
	dtoNNs[i] = temp;
      }
    }
  }
  
  free(dtoNNs);  
}
#endif
