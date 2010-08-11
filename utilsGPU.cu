#ifndef UTILSGPU_CU
#define UTILSGPU_CU

#include<cuda.h>
#include<stdio.h>
#include "defs.h"

memPlan createMemPlan(int nPts, int memPerPt){
  memPlan mp;
  unsigned int memFree, memTot;
  int ptsAtOnce;

  cuMemGetInfo(&memFree, &memTot);
  memFree = (int)(((float)memFree)*MEM_USABLE);
  printf("memfree = %d \n",memFree);
  ptsAtOnce = DPAD(memFree/memPerPt); //max number of pts that can be processed at once
  printf("ptsAtOnce = %d \n",ptsAtOnce);
  mp.numComputeSegs = nPts/ptsAtOnce + ((nPts%ptsAtOnce==0) ? 0 : 1);
  mp.normSegSize=PAD(nPts/mp.numComputeSegs); 
  mp.lastSegSize=PAD(nPts) - mp.normSegSize*(mp.numComputeSegs-1);
  //Note that lastSegSize is automatically padded if nPts is.
  return mp;
}

void copyAndMove(matrix *dx, const matrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}


void copyAndMoveI(intMatrix *dx, const intMatrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}


void copyAndMoveC(charMatrix *dx, const charMatrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}
#endif
