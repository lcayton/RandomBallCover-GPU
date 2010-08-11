/* ****** Device helper functions ****** */

__device__ matrix getSubMat(matrix A, int row, int col){
  matrix As;
  
  As.c = BLOCK_SIZE;
  As.r = BLOCK_SIZE;
  As.ld = A.ld;
  As.mat = &A.mat[A.ld*row*BLOCK_SIZE + col*BLOCK_SIZE];
  
  return As;
}



//l_1-norm
/* __global__ void dist1Kernel(const matrix Q, const matrix X, matrix D){ */
/*   int qBlockRow = blockIdx.y; */
/*   int xBlockRow = blockIdx.x; */

/*   //matrix Dsub = getSubMat(D,xBlockRow,qBlockRow); */

/*   int qID = threadIdx.y; */
/*   int xID = threadIdx.x; */

/*   //  printf("calling (%d,%d) \n",qBlockRow*BLOCK_SIZE+qID,xBlockRow*BLOCK_SIZE+xID); */
  
/*   int i,j; */
/*   real ans=0; */
  
/*   //Note: assumes that X is padded. */
/*   for(i=0;i<X.pc/BLOCK_SIZE;i++){ */
/*     //    matrix Xs = getSubMat(X,xBlockRow,i); */
/*     //    matrix Qs = getSubMat(Q,qBlockRow,i); */
    
/*     __shared__ real Xb[BLOCK_SIZE][BLOCK_SIZE]; */
/*     __shared__ real Qb[BLOCK_SIZE][BLOCK_SIZE]; */
    
/*     // Each thread loads one element of Xs and Qs into shared mem */
/*     // Note that the indexing is swapped to increase memory coalescing. */
/*     //printf("reading x[%d,%d] \n",xBlockRow*BLOCK_SIZE+qID,i*BLOCK_SIZE+xID); */
/*     Xb[xID][qID]=X.mat[IDX(xBlockRow*BLOCK_SIZE+qID,i*BLOCK_SIZE+xID,X.ld)];// getElement(Xs,qID,xID); */
/*     //printf("reading q[%d,%d] \n",qBlockRow*BLOCK_SIZE+qID,i*BLOCK_SIZE+xID); */
/*     Qb[xID][qID]=Q.mat[IDX(qBlockRow*BLOCK_SIZE+qID,i*BLOCK_SIZE+xID,Q.ld)]; */
    
/*     __syncthreads(); */
    
/*     for(j=0;j<BLOCK_SIZE;j++) */
/*       ans+=abs(Xb[j][xID]-Qb[j][qID]); */
        
/*       __syncthreads(); */
/*   } */
  
/*   //  Dsub.mat[IDX(qID,xID,Dsub.ld)]=ans;//setElement(Dsub,qID,xID,ans); */
/*   //printf("writing (%d,%d) %6.2f \n",qBlockRow*BLOCK_SIZE+qID,xBlockRow*BLOCK_SIZE+xID,ans); */
/*   //  Dsub.mat[IDX(qID,xID,Dsub.ld)]=ans; */
/*   D.mat[IDX(qBlockRow*BLOCK_SIZE+qID,xBlockRow*BLOCK_SIZE+xID,D.ld)]=ans; */
  
/* } */
