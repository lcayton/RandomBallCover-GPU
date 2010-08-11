#ifndef SKERNEL_CU
#define SKERNEL_CU

#include<stdio.h>
#include "sKernel.h"
#include "defs.h"

__global__ void sumKernel(charMatrix in, intMatrix sum, intMatrix sumaux, int n){
  int id = threadIdx.x;
  int bo = blockIdx.x*SCAN_WIDTH; //block offset
  int r = blockIdx.y;
  int d, t;
  
  const int l=SCAN_WIDTH; //length

  int off=1;

  __shared__ int ssum[l];

  
  ssum[2*id] = (bo+2*id < n) ? in.mat[IDX( r, bo+2*id, in.ld )] : 0;
  ssum[2*id+1] = (bo+2*id+1 < n) ? in.mat[IDX( r, bo+2*id+1, in.ld)] : 0;
    
  //up-sweep
  for( d=l>>1; d > 0; d>>=1 ){
    __syncthreads();
    
    if( id < d ){
      ssum[ off*(2*id+2)-1 ] += ssum[ off*(2*id+1)-1 ];
    }
    off *= 2;
  }
  
  __syncthreads();
  
  if ( id == 0 ){
    sumaux.mat[IDX( r, blockIdx.x, sumaux.ld )] = ssum[ l-1 ];
    ssum[ l-1 ] = 0;
  }
  
  //down-sweep
  for ( d=1; d<l; d*=2 ){
    off >>= 1;
    __syncthreads();
    
    if( id < d ){
      t = ssum[ off*(2*id+1)-1 ];
      ssum[ off*(2*id+1)-1 ] = ssum[ off*(2*id+2)-1 ];
      ssum[ off*(2*id+2)-1 ] += t;
    }
    
  }

  __syncthreads();
 
  if( bo+2*id < n ) 
    sum.mat[IDX( r, bo+2*id, sum.ld )] = ssum[2*id];
  if( bo+2*id+1 < n )
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] = ssum[2*id+1];

}

__global__ void combineSumKernel(intMatrix sum, intMatrix daux, int n){
  int id = threadIdx.x;
  int bo = blockIdx.x * SCAN_WIDTH;
  int r = blockIdx.y;

  if(bo+2*id < n)
    sum.mat[IDX( r, bo+2*id, sum.ld )] += daux.mat[IDX( r, blockIdx.x, daux.ld )];
  if(bo+2*id+1 < n)
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] += daux.mat[IDX( r, blockIdx.x, daux.ld )];
  
}


__global__ void getCountsKernel(int *counts, charMatrix ir, intMatrix sums){
  int r = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if ( r < ir.r )
    counts[r] = ir.mat[IDX( r, ir.c-1, ir.ld )] ? sums.mat[IDX( r, sums.c-1, sums.ld )]+1 : sums.mat[IDX( r, sums.c-1, sums.ld )];
}


__global__ void buildMapKernel(intMatrix map, charMatrix ir, intMatrix sums, int offSet){
  int id = threadIdx.x;
  int bo = blockIdx.x * SCAN_WIDTH;
  int r = blockIdx.y;

  if(bo+2*id < ir.c && ir.mat[IDX( r, bo+2*id, ir.ld )])
    map.mat[IDX( r+offSet, sums.mat[IDX( r, bo+2*id, sums.ld )], map.ld)] = bo+2*id;
  if(bo+2*id+1 < ir.c && ir.mat[IDX( r, bo+2*id+1, ir.ld )])
    map.mat[IDX( r+offSet, sums.mat[IDX( r, bo+2*id+1, sums.ld )], map.ld)] = bo+2*id+1;
}


void getCountsWrap(int *counts, charMatrix ir, intMatrix sums){
  dim3 block(BLOCK_SIZE,1);
  dim3 grid(ir.pr/BLOCK_SIZE,1);
  getCountsKernel<<<grid,block>>>(counts, ir, sums);

}


void buildMapWrap(intMatrix map, charMatrix ir, intMatrix sums, int offSet){
  int numScans = (ir.c+SCAN_WIDTH-1)/SCAN_WIDTH;
  dim3 block( SCAN_WIDTH/2, 1 );
  dim3 grid( numScans, ir.r );
  buildMapKernel<<<grid,block>>>(map, ir, sums, offSet);
}


void sumWrap(charMatrix in, intMatrix sum){
  int n = in.c;
  int numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;

  if(numScans > SCAN_WIDTH){
    fprintf(stderr,"scan is too large.. exiting \n");
    exit(1);
  }
  
  intMatrix dAux;
  dAux.r=dAux.pr=in.r; dAux.c=dAux.pc=dAux.ld=numScans;
  intMatrix dDummy;
  dDummy.r=dDummy.pr=in.r; dDummy.c=dDummy.pc=dDummy.ld=1;
  
  cudaMalloc( (void**)&dAux.mat, dAux.pr*dAux.pc*sizeof(*dAux.mat) );
  cudaMalloc( (void**)&dDummy.mat, dDummy.pr*dDummy.pc*sizeof(*dDummy.mat) );

  dim3 block( SCAN_WIDTH/2, 1 );
  dim3 grid( numScans, in.r );
  
  sumKernel<<<grid,block>>>(in, sum, dAux, n);
  cudaThreadSynchronize();
  grid.x=1;

  sumKernelI<<<grid,block>>>(dAux, dAux, dDummy, numScans);
  cudaThreadSynchronize();

  grid.x=numScans;
  combineSumKernel<<<grid,block>>>(sum,dAux,n);
  cudaThreadSynchronize();

  cudaFree(dAux.mat);
  cudaFree(dDummy.mat);

}


//This is the same as sumKernel, but takes an int matrix as input.
__global__ void sumKernelI(intMatrix in, intMatrix sum, intMatrix sumaux, int n){
  int id = threadIdx.x;
  int bo = blockIdx.x*SCAN_WIDTH; //block offset
  int r = blockIdx.y;
  int d, t;
  
  const int l=SCAN_WIDTH; //length

  int off=1;

  __shared__ int ssum[l];

  
  ssum[2*id] = (bo+2*id < n) ? in.mat[IDX( r, bo+2*id, in.ld )] : 0;
  ssum[2*id+1] = (bo+2*id+1 < n) ? in.mat[IDX( r, bo+2*id+1, in.ld)] : 0;
    
  //up-sweep
  for( d=l>>1; d > 0; d>>=1 ){
    __syncthreads();
    
    if( id < d ){
      ssum[ off*(2*id+2)-1 ] += ssum[ off*(2*id+1)-1 ];
    }
    off *= 2;
  }
  
  __syncthreads();
  
  if ( id == 0 ){
    sumaux.mat[IDX( r, blockIdx.x, sumaux.ld )] = ssum[ l-1 ];
    ssum[ l-1 ] = 0;
  }
  
  //down-sweep
  for ( d=1; d<l; d*=2 ){
    off >>= 1;
    __syncthreads();
    
    if( id < d ){
      t = ssum[ off*(2*id+1)-1 ];
      ssum[ off*(2*id+1)-1 ] = ssum[ off*(2*id+2)-1 ];
      ssum[ off*(2*id+2)-1 ] += t;
    }
    
  }

  __syncthreads();
 
  if( bo+2*id < n ) 
    sum.mat[IDX( r, bo+2*id, sum.ld )] = ssum[2*id];
  if( bo+2*id+1 < n )
    sum.mat[IDX( r, bo+2*id+1, sum.ld )] = ssum[2*id+1];

}

#endif
