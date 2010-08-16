#ifndef KERNELWRAP_CU
#define KERNELWRAP_CU

#include<cuda.h>
#include<stdio.h>
#include "kernels.h"
#include "defs.h"

void dist1Wrap(matrix dq, matrix dx, matrix dD){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid;
  
  int todoX, todoY, numDoneX, numDoneY;

  numDoneX = 0;
  while ( numDoneX < dx.pr ){
    todoX = min( dx.pr - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dq.pr ){
      todoY = min( dq.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      dist1Kernel<<<grid,block>>>(dq, numDoneY, dx, numDoneX, dD);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}


void findRangeWrap(matrix dD, real *dranges, int cntWant){
  dim3 block(4*BLOCK_SIZE,BLOCK_SIZE/4);
  dim3 grid(1,4*(dD.pr/BLOCK_SIZE));

  findRangeKernel<<<grid,block>>>(dD,dranges,cntWant);

  
  cudaThreadSynchronize();
}

void rangeSearchWrap(matrix dD, real *dranges, charMatrix dir){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(dD.pc/BLOCK_SIZE,dD.pr/BLOCK_SIZE);

  int todoX, todoY, numDoneX, numDoneY;
  
  numDoneX = 0;
  while ( numDoneX < dD.pc ){
    todoX = min( dD.pc - numDoneX, MAX_BS*BLOCK_SIZE );
    grid.x = todoX/BLOCK_SIZE;
    numDoneY = 0;
    while( numDoneY < dD.pr ){
      todoY = min( dD.pr - numDoneY, MAX_BS*BLOCK_SIZE );
      grid.y = todoY/BLOCK_SIZE;
      rangeSearchKernel<<<grid,block>>>(dD, numDoneX, numDoneY, dranges, dir);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}

void nnWrap(const matrix dx, const matrix dy, real *dMins, int *dMinIDs){
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid;
  
  dimGrid.x = 1;
  dimGrid.y = dx.pr/dimBlock.y + (dx.pr%dimBlock.y==0 ? 0 : 1);
  nnKernel<<<dimGrid,dimBlock>>>(dx,dy,dMins,dMinIDs);
  cudaThreadSynchronize();
}


void rangeCountWrap(const matrix dq, const matrix dx, real *dranges, int *dcounts){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(1,dq.pr/BLOCK_SIZE);

  rangeCountKernel<<<grid,block>>>(dq,dx,dranges,dcounts);
  cudaThreadSynchronize();
}


/*NOTE: can be deleted */
void pruneWrap(charMatrix dcM, matrix dD, real *dradiiX, real *dradiiQ){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(dD.pr/BLOCK_SIZE,dD.pc/BLOCK_SIZE);
  
  pruneKernel<<<grid,block>>>(dD,dradiiX,dradiiQ,dcM);
  cudaThreadSynchronize();
}
#endif
