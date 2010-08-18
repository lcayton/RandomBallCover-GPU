/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef KERNELWRAP_CU
#define KERNELWRAP_CU

#include<cuda.h>
#include<stdio.h>
#include "kernels.h"
#include "defs.h"

void dist1Wrap(const matrix dq, const matrix dx, matrix dD){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  
  unint todoX, todoY, numDoneX, numDoneY;

  numDoneX = 0;
  while ( numDoneX < dx.pr ){
    todoX = MIN( dx.pr - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dq.pr ){
      todoY = MIN( dq.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      dist1Kernel<<<grid,block>>>(dq, numDoneY, dx, numDoneX, dD);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}


void findRangeWrap(const matrix dD, real *dranges, unint cntWant){
  dim3 block(4*BLOCK_SIZE,BLOCK_SIZE/4);
  dim3 grid(1,4*(dD.pr/BLOCK_SIZE));
  unint numDone, todo;
  
  numDone=0;
  while( numDone < dD.pr ){
    todo = MIN ( dD.pr - numDone, MAX_BS*BLOCK_SIZE/4 );
    grid.y = 4*(todo/BLOCK_SIZE);
    findRangeKernel<<<grid,block>>>(dD, numDone, dranges, cntWant);
    numDone += todo;
  }
  cudaThreadSynchronize();
}


void rangeSearchWrap(const matrix dD, const real *dranges, charMatrix dir){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;

  unint todoX, todoY, numDoneX, numDoneY;
  
  numDoneX = 0;
  while ( numDoneX < dD.pc ){
    todoX = MIN( dD.pc - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dD.pr ){
      todoY = MIN( dD.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      rangeSearchKernel<<<grid,block>>>(dD, numDoneX, numDoneY, dranges, dir);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}

void nnWrap(const matrix dq, const matrix dx, real *dMins, unint *dMinIDs){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint numDone, todo;
  
  grid.x = 1;

  numDone = 0;
  while( numDone < dq.pr ){
    todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    nnKernel<<<grid,block>>>(dq,numDone,dx,dMins,dMinIDs);
    numDone += todo;
  }
  cudaThreadSynchronize();

}


void planNNWrap(const matrix dq, const unint *dqMap, const matrix dx, const intMatrix dxMap, real *dMins, unint *dMinIDs, compPlan dcP, unint compLength){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint todo;

  grid.x = 1;
  unint numDone = 0;
  while( numDone<compLength ){
    todo = MIN( (compLength-numDone) , MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    planNNKernel<<<grid,block>>>(dq,dqMap,dx,dxMap,dMins,dMinIDs,dcP,numDone);
    numDone += todo;
  }
  cudaThreadSynchronize();
}


void rangeCountWrap(const matrix dq, const matrix dx, real *dranges, unint *dcounts){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  unint numDone, todo;

  grid.x=1;

  numDone = 0;
  while( numDone < dq.pr ){
    todo = MIN( dq.pr - numDone, MAX_BS*BLOCK_SIZE );
    grid.y = todo/BLOCK_SIZE;
    rangeCountKernel<<<grid,block>>>(dq,numDone,dx,dranges,dcounts);
    numDone += todo;
  }
  cudaThreadSynchronize();
}

#endif
