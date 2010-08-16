#ifndef SKERNEL_CU
#define SKERNEL_CU

#include<stdio.h>
#include<math.h>
#include "sKernel.h"
#include "defs.h"
#include "utils.h"

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
  if ( r < ir.r ){
    counts[r] = ir.mat[IDX( r, ir.c-1, ir.ld )] ? sums.mat[IDX( r, sums.c-1, sums.ld )]+1 : sums.mat[IDX( r, sums.c-1, sums.ld )];
  }
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
  int i;
  int n = in.c;
  int numScans = (n+SCAN_WIDTH-1)/SCAN_WIDTH;
  

  int depth = ceil( log(n) / log(SCAN_WIDTH) ) -1 ;
  int *width = (int*)calloc( depth+1, sizeof(*width) );
    
  intMatrix *dAux;
  dAux = (intMatrix*)calloc( depth+1, sizeof(*dAux) );
  
  dAux[0].r=dAux[0].pr=in.r; dAux[0].c=dAux[0].pc=dAux[0].ld=numScans;
  cudaMalloc( (void**)&dAux[0].mat, dAux[0].pr*dAux[0].pc*sizeof(*dAux[0].mat) );
   
  dim3 block( SCAN_WIDTH/2, 1 );
  dim3 grid( numScans, in.r );
  
  sumKernel<<<grid,block>>>(in, sum, dAux[0], n);
  cudaThreadSynchronize();
  
  width[0]=numScans; //Clean up, this is ugly (necc b/c loop might not be entered)
  for( i=0; i<depth; i++ ){
    width[i] = numScans;
    numScans = (numScans+SCAN_WIDTH-1)/SCAN_WIDTH;
    
    dAux[i+1].r=dAux[i+1].pr=in.r; dAux[i+1].c=dAux[i+1].pc=dAux[i+1].ld=numScans;
    cudaMalloc( (void**)&dAux[i+1].mat, dAux[i+1].pr*dAux[i+1].pc*sizeof(*dAux[i+1].mat) );
        
    grid.x = numScans;
    sumKernelI<<<grid,block>>>(dAux[i], dAux[i], dAux[i+1], width[i]);
    cudaThreadSynchronize();
  }

    
  for( i=depth-1; i>0; i-- ){
    grid.x = width[i];
    combineSumKernel<<<grid,block>>>(dAux[i-1], dAux[i], width[i-1]);
    cudaThreadSynchronize();
  }
  
  grid.x = width[0];
  combineSumKernel<<<grid,block>>>(sum, dAux[0], n);
  cudaThreadSynchronize();
  
  for( i=0; i <= depth; i++)
    cudaFree(dAux[i].mat);
  
  free(dAux);
  free(width);
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
