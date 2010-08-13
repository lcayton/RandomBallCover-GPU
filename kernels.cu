#ifndef KERNELS_CU
#define KERNELS_CU

#include<cuda.h>
#include "defs.h"
#include "kernels.h"
#include<stdio.h>

// This kernel does the same thing as nnKernel, except it only considers pairs as 
// specified by the compPlan.
__global__ void planNNKernel(const matrix Q, const matrix X, real *dMins, int *dMinIDs, compPlan cP, intMatrix xmap, int *qIDs, int qStartPos ){
  int qBlock = qStartPos + blockIdx.y * BLOCK_SIZE;  //indexes Q
  int xBlock; //indexes X;
  int colBlock;
  int offQ = threadIdx.y; //the offset of qPos in this block
  int offX = threadIdx.x; //ditto for x
  int i,j,k,l;
  
  
  __shared__ real min[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int minPos[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real Xb[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qb[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ int g; //group of q  
  __shared__ int numGroups;
  __shared__ int groupCount;
  __shared__ int groupOff;

  if(offX==0 && offQ==0){
    g = cP.qToGroup[qBlock]; 
    numGroups = cP.numGroups[g];
  }
  min[offQ][offX]=MAX_REAL;
  __syncthreads();
      
  //NOTE: might be better to save numGroup, groupIts in local reg.

  for(i=0;i<numGroups;i++){ //iterate over groups of X  
    if(offX==0 && offQ==0){
      groupCount = cP.groupCountX[IDX(g,i,cP.ld)];
      groupOff = cP.groupOff[IDX(g,i,cP.ld)];
    }

    __syncthreads();
    
    int groupIts = groupCount/BLOCK_SIZE + (groupCount%BLOCK_SIZE==0? 0 : 1);
   
    xBlock=groupOff;
    for(j=0;j<groupIts;j++){ //iterate over elements of group
      xBlock=j*BLOCK_SIZE;
      
      real ans=0;
      for(k=0;k<X.pc/BLOCK_SIZE;k++){ // iterate over cols to compute the distances
	colBlock = k*BLOCK_SIZE;

	//Each thread loads one element of X and Q into memory.
	//Note that the indexing is flipped to increase memory
	//coalescing.

	Xb[offX][offQ] = X.mat[IDX( xmap.mat[IDX( g, xBlock+offQ, xmap.ld)], colBlock+offX, X.ld)];
	Qb[offX][offQ] = ( (qIDs[qBlock+offQ]==DUMMY_IDX) ? 0 : Q.mat[IDX(qIDs[qBlock+offQ],colBlock+offX,Q.ld)] );
	__syncthreads();
	
	for(l=0;l<BLOCK_SIZE;l++){
	  ans+=abs(Xb[l][offX]-Qb[l][offQ]);
	}
	__syncthreads();
      }
      
      //compare to previous min and store into shared mem if needed.
      if(j*BLOCK_SIZE+offX<groupCount && ans<min[offQ][offX]){
	min[offQ][offX]=ans;
	minPos[offQ][offX]= xmap.mat[IDX( g, xBlock+offX, xmap.ld )];
      }
      __syncthreads();
      
    }
  }
  
  //compare across threads
  for(k=BLOCK_SIZE/2;k>0;k/=2){
    if(offX<k){
      if(min[offQ][offX+k]<min[offQ][offX]){
	min[offQ][offX] = min[offQ][offX+k];
	minPos[offQ][offX] = minPos[offQ][offX+k];	
      }
    }
    __syncthreads();
  }
  
  if(offX==0 && qIDs[qBlock+offQ]!=DUMMY_IDX){
    dMins[qIDs[qBlock+offQ]] = min[offQ][0];
    dMinIDs[qIDs[qBlock+offQ]] = minPos[offQ][0];
  }
}



__global__ void pruneKernel(const matrix D, const real *radiiX, const real *radiiQ, charMatrix cM){
  int offX = threadIdx.x;
  int offQ = threadIdx.y;

  int blockX = blockIdx.x * BLOCK_SIZE;
  int blockQ = blockIdx.y * BLOCK_SIZE;
  
  __shared__ real sD[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real sRQ[BLOCK_SIZE];
  __shared__ real sRX[BLOCK_SIZE];

  sD[offQ][offX]=D.mat[IDX(blockQ+offQ,blockX+offX,D.ld)];
  
  if(offQ==0)
    sRX[offX]=radiiX[blockX+offX];
  if(offX==0)
    sRQ[offQ]=radiiQ[blockQ+offQ];
  
  __syncthreads();
  
  if(blockQ+offQ < D.r && blockX+offX < D.c){
    cM.mat[IDX(blockQ+offQ,blockX+offX,cM.ld)] = (sD[offQ][offX]-sRX[offX]-2*sRQ[offQ] <= 0) ? 1 : 0;
    //cM.mat[IDX(blockQ+offQ,blockX+offX,cM.ld)] = (sD[offQ][offX]-4*sRQ[offQ] <= 0) ? 1 : 0;
  }
}


__global__ void nnKernel(const matrix Q, const matrix X, real *dMins, int *dMinIDs){

  int qBlock = blockIdx.y * BLOCK_SIZE;  //indexes Q
  int xBlock; //indexes X;
  int colBlock;
  int offQ = threadIdx.y; //the offset of qPos in this block
  int offX = threadIdx.x; //ditto for x
  int i,j,k;

  __shared__ real min[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int minPos[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real Xb[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qb[BLOCK_SIZE][BLOCK_SIZE];

  min[offQ][offX]=MAX_REAL;
  __syncthreads();

  for(i=0;i<X.pr/BLOCK_SIZE;i++){
    xBlock = i*BLOCK_SIZE;
    real ans=0;
    for(j=0;j<X.pc/BLOCK_SIZE;j++){
      colBlock = j*BLOCK_SIZE;
      
      //Each thread loads one element of X and Q into memory.
      //Note that the indexing is flipped to increase memory
      //coalescing.
      Xb[offX][offQ]=X.mat[IDX(xBlock+offQ,colBlock+offX,X.ld)];
      Qb[offX][offQ]=Q.mat[IDX(qBlock+offQ,colBlock+offX,Q.ld)];

      __syncthreads();

      for(k=0;k<BLOCK_SIZE;k++){
	ans+=abs(Xb[k][offX]-Qb[k][offQ]);
      }
      __syncthreads();
    }
   
    
    if( xBlock+offX<X.r && ans<min[offQ][offX] ){
       minPos[offQ][offX] = xBlock+offX;
       min[offQ][offX] = ans;
    }
  }
  __syncthreads();
  
  
  //reduce across threads
  for(j=BLOCK_SIZE/2;j>0;j/=2){
    if(offX<j){
      if(min[offQ][offX+j]<min[offQ][offX]){
	min[offQ][offX] = min[offQ][offX+j];
	minPos[offQ][offX] = minPos[offQ][offX+j];	
      }
    }
    __syncthreads();
  }
  
  if(offX==0){
    //printf("writing %d, nn= %d, val = %6.4f \n",qBlock+offQ,curMinPos[offQ],curMin[offQ]);
    dMins[qBlock+offQ] = min[offQ][0];
    dMinIDs[qBlock+offQ] = minPos[offQ][0];
  }
}


__device__ void dist1Kernel(const matrix Q, int qStart, const matrix X, int xStart, matrix D){
  int c, i, j;

  int qB = blockIdx.y*BLOCK_SIZE + qStart;
  int q  = threadIdx.y;
  int xB = blockIdx.x*BLOCK_SIZE + xStart;
  int x = threadIdx.x;

  real ans=0;

  //This thread is responsible for computing the dist between Q[qB+q] and X[xB+x]
  
  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];


  for(i=0 ; i<Q.pc/BLOCK_SIZE ; i++){
    c=i*BLOCK_SIZE; //current col block

    Qs[x][q] = Q.mat[ IDX(qB+q, c+x, Q.ld) ];
    Xs[x][q] = X.mat[ IDX(xB+q, c+x, X.ld) ];

    __syncthreads();

    for(j=0 ; j<BLOCK_SIZE ; j++)
      ans += abs( Qs[j][q] - Xs[j][x] );
        
    __syncthreads();
  }
  
  D.mat[ IDX( qB+q, xB+x, D.ld ) ] = ans;

}



__global__ void findRangeKernel(matrix D, real *ranges, int cntWant){
  
  int row = blockIdx.y*(BLOCK_SIZE/4)+threadIdx.y;
  int ro = threadIdx.y;
  int co = threadIdx.x;
  int i, c;
  real t;

  const int LB = (90*cntWant)/100 ;
  const int UB = cntWant; 

  __shared__ real smin[BLOCK_SIZE/4][4*BLOCK_SIZE];
  __shared__ real smax[BLOCK_SIZE/4][4*BLOCK_SIZE];
  
  real min=MAX_REAL;
  real max=0;
  for(c=0 ; c<D.pc ; c+=(4*BLOCK_SIZE)){
    if( c+co < D.c ){
      t = D.mat[ IDX( row, c+co, D.ld ) ];
      min = MIN(t,min);
      max = MAX(t,max);
    }
  }
  
  smin[ro][co] = min;
  smax[ro][co] = max;
  __syncthreads();
  
  for(i=2*BLOCK_SIZE ; i>0 ; i/=2){
    if( co < i ){
      smin[ro][co] = MIN( smin[ro][co], smin[ro][co+i] );
      smax[ro][co] = MAX( smax[ro][co], smax[ro][co+i] );
    }
    __syncthreads();
  }

  //Now start range counting.

  int itcount=0;
  int cnt;
  real rg;
  __shared__ int scnt[BLOCK_SIZE/4][4*BLOCK_SIZE];
  __shared__ char cont[BLOCK_SIZE/4];
  
  if(co==0)
    cont[ro]=1;
  
  do{
    itcount++;
    __syncthreads();

    if( cont[ro] )  //if we didn't actually need to cont, leave rg as it was.
      rg = ( smax[ro][0] + smin[ro][0] ) / ((real)2.0) ;

    cnt=0;
    for(c=0 ; c<D.pc ; c+=(4*BLOCK_SIZE)){
      cnt += (c+co < D.c && row < D.r && D.mat[ IDX( row, c+co, D.ld ) ] <= rg);
    }

    scnt[ro][co] = cnt;
    __syncthreads();
    
    for(i=2*BLOCK_SIZE ; i>0 ; i/=2){
      if( co < i ){
	scnt[ro][co] += scnt[ro][co+i];
      }
      __syncthreads();
    }
    
    if(co==0){
      if( scnt[ro][0] < cntWant )
	smin[ro][0]=rg;
      else
	smax[ro][0]=rg;
    }
    
    // cont[ro] == this row needs to continue
    if(co==0)
      cont[ro] = row<D.r && ( scnt[ro][0] < LB || scnt[ro][0] > UB ); 
    __syncthreads();

    // Determine if *any* of the rows need to continue
    for(i=BLOCK_SIZE/8 ; i>0 ; i/=2){
      if( ro < i && co==0)
	cont[ro] |= cont[ro+i];
      __syncthreads();
    }
    
  } while(cont[0]);

  if(co==0 && row<D.r )
    ranges[row]=rg;
  
}


__global__ void rangeSearchKernel(matrix D, int xOff, int yOff, real *ranges, charMatrix ir){
  int col = blockIdx.x*BLOCK_SIZE + threadIdx.x + xOff;
  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y + yOff;

  ir.mat[IDX( row, col, ir.ld )] = D.mat[IDX( row, col, D.ld )] < ranges[row];

}


__global__ void rangeCountKernel(const matrix Q, const matrix X, real *ranges, int *counts){
  int q = blockIdx.y*BLOCK_SIZE;
  int qo = threadIdx.y;
  int xo = threadIdx.x;
  
  real rg = ranges[q+qo];
  
  int r,c,i;

  __shared__ int scnt[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real qs[BLOCK_SIZE][BLOCK_SIZE];
  
  int cnt=0;
  for( r=0; r<X.pr; r+=BLOCK_SIZE ){

    real dist=0;
    for( c=0; c<X.pc; c+=BLOCK_SIZE){
      xs[xo][qo] = X.mat[IDX( r+qo, c+xo, X.ld )];
      qs[xo][qo] = Q.mat[IDX( q+qo, c+xo, Q.ld )];
      __syncthreads();
      
      for( i=0; i<BLOCK_SIZE; i++)
	dist += abs(xs[i][xo]-qs[i][qo]);
      __syncthreads();

    }
    cnt += r+xo<X.r && dist<rg;

  }
  
  scnt[qo][xo]=cnt;
  __syncthreads();
  
  for( i=BLOCK_SIZE/2; i>0; i/=2 ){
    if( xo<i ){
      scnt[qo][xo] += scnt[qo][xo+i];
    }
    __syncthreads();
  }

  if( xo==0 && q+qo<Q.r )
    counts[q+qo] = scnt[qo][0];
}


#endif
