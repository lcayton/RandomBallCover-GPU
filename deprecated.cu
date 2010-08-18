/* This file contains functions that were in use at one point but 
   are not currently used.  As far as I know, everything here is
   debugged, and so can be plugged in reasonably safely.
*/



/* This is the more complex version of computeReps.  It can be used even if X doesn't
fit into the device memory.  It does not work in emulation mode, because it checks to see
how much mem is available on the device.  Thus for debugging purposes we currently 
use a simpler version of computeReps. */

//Assumes that dr is a matrix already on the device
void computeReps(matrix x, matrix dr, int *repIDs, real *distToReps){
  int memPerX, segSize; //seg==segment
  int index, tempSize; //temp variables used in the loop
  int i;
  matrix dx;
  real *dMins;
  int *dMinIDs;
  memPlan mp;

  int n = x.r; //For convenience
  
  // Items that need to go on device: x, repIDs, distToReps.  The "+1" is for the
  // distance from each point to its nearest rep (distToReps) and the int is for
  // the ID (repIDs).
  memPerX = (x.pc+1)*sizeof(real)+sizeof(int);
  mp = createMemPlan(x.r,memPerX);
  
  for(i=0;i<mp.numComputeSegs;i++){
    if(i==mp.numComputeSegs-1)
      segSize = mp.lastSegSize;
    else
      segSize = mp.normSegSize;

    //Allocate & copy over data
    index = IDX(mp.normSegSize*i,0,x.ld);
    tempSize = segSize*x.pc*sizeof(*(dx.mat));

    cudaMalloc((void**)&(dx.mat),tempSize);
    cudaMemcpy(dx.mat,&(x.mat[index]),tempSize,cudaMemcpyHostToDevice);
    dx.r=segSize; dx.c=x.c; dx.pr=dx.r; dx.pc=x.pc; dx.ld=x.ld;

    //Allocate matrices to temporarily store mins and IDs (NOTE:MOVE OUT OF LOOP FOR EFFICIENCY)
    cudaMalloc((void**)&(dMins), PAD(MIN(segSize,n))*sizeof(*dMins));
    cudaMalloc((void**)&(dMinIDs), PAD(MIN(segSize,n))*sizeof(*dMinIDs));
    nnWrap(dx,dr,dMins,dMinIDs);

    cudaMemcpy(&distToReps[i*segSize],dMins,MIN(segSize,n)*sizeof(*dMins),cudaMemcpyDeviceToHost);
    cudaMemcpy(&repIDs[i*segSize],dMinIDs,MIN(segSize,n)*sizeof(*dMinIDs),cudaMemcpyDeviceToHost);
    
    cudaFree(dMins);
    cudaFree(dMinIDs);
    cudaFree(dx.mat);
  }
}


__global__ void getMinsKernel(matrix,real*,int*);

// Returns the min of each row of D.  dMins and dMinIDs 
// are assumed to be (at least) of size D.r.
__global__ void getMinsKernel(const matrix D, real *dMins, int *dMinIDs){
  int row, locRow, colOff, i, curCol;
  real temp;

  row = blockIdx.y*BLOCK_SIZE+threadIdx.y;
  locRow = threadIdx.y;
  
  colOff = threadIdx.x; //column offset of this thread
 
  __shared__ float mins[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int pos[BLOCK_SIZE][BLOCK_SIZE];
  

  // This loop finds the minimum of cols 
  // [colOff, BLOCK_SIZE+colOff, 2*BLOCK_SIZE+colOff,...]
  // and stores it in mins[locRow][colOff].
  mins[locRow][colOff]=MAX_REAL;
  pos[locRow][colOff]=-1;
  for (i=0;i<D.pc/BLOCK_SIZE;i++){
    curCol = i*BLOCK_SIZE+colOff;
    if(curCol < D.c){ //ignore padding
      temp = D.mat[IDX(row,curCol,D.ld)];
      if(temp<mins[locRow][colOff]){
	mins[locRow][colOff]=temp;
	pos[locRow][colOff]=curCol;
      }
    }
  }
  __syncthreads();
    
  //Now find the min of cols [0, ... , BLOCK_SIZE]
  for (i=BLOCK_SIZE/2; i>0;i/=2){
    if(colOff<i){
      //compare (col) to (col+i)
      if(mins[locRow][colOff]>mins[locRow][colOff+i]){
	mins[locRow][colOff]=mins[locRow][colOff+i];
	pos[locRow][colOff]=pos[locRow][colOff+i];
      }
    }
    __syncthreads();
  }
  
  //arbitrarily use the first thread (along x) to set memory
  if(threadIdx.x==0){  
    dMins[row] = mins[locRow][0];
    dMinIDs[row] = pos[locRow][0];
  }
}


// Returns the min of each row of D.  dMins and dMinIDs 
// are assumed to be (at least) of size D.r.
__global__ void getKMinsKernel(matrix D, matrix dMins, intMatrix NNs, int k){
  int row, locRow, colOff, i, curCol,j;
  real temp;

  row = blockIdx.y*BLOCK_SIZE+threadIdx.y;
  locRow = threadIdx.y;

  //printf("row=%d D.r =%d \n",row,D.r);
  /* if(row>=D.r) */
  /*   return; */

  colOff = threadIdx.x; //column offset of this thread
 
  __shared__ float mins[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int pos[BLOCK_SIZE][BLOCK_SIZE];
  
  for(i=0;i<k;i++){
    // This loop finds the minimum of cols 
    // [colOff, BLOCK_SIZE+colOff, 2*BLOCK_SIZE+colOff,...]
    // and stores it in mins[locRow][colOff].
    mins[locRow][colOff]=MAX_REAL;
    pos[locRow][colOff]=-1;
    for (j=0;j<D.pc/BLOCK_SIZE;j++){
      curCol = j*BLOCK_SIZE+colOff;
      if(curCol < D.c){ //ignore padding
	temp = D.mat[IDX(row,curCol,D.ld)];
	if(temp<mins[locRow][colOff]){
	  mins[locRow][colOff]=temp;
	  pos[locRow][colOff]=curCol;
	}
      }
    }
    __syncthreads();


    //Now find the min of cols [0, ... , BLOCK_SIZE]
    for (j=BLOCK_SIZE/2; j>0; j/=2){
      if(colOff<j){	
	//compare (col) to (col+j)
	if(mins[locRow][colOff]>mins[locRow][colOff+j]){
	  mins[locRow][colOff]=mins[locRow][colOff+j];
	  pos[locRow][colOff]=pos[locRow][colOff+j];
	}
      }
       __syncthreads();
    }
    
  //arbitrarily use the first thread (along x) to set memory
    if(threadIdx.x==0 && row<D.r){  
      dMins.mat[IDX(row,i,dMins.ld)] = mins[locRow][0];
      NNs.mat[IDX(row,i,NNs.ld)] = pos[locRow][0];
      D.mat[IDX(row,pos[locRow][0],D.ld)]=MAX_REAL;
      
    }
    __syncthreads();
  }
}

size_t countCompute(int*,int*,charMatrix);

//This is used for debugging/research.
size_t countCompute(int *groupCountQ, int *groupCountX, charMatrix cM){
  int i,j;
  size_t ans=0;
  size_t avgBlocks=0;
  size_t maxBlocks=0;
  size_t maxBlocksInd;
  size_t maxTemp;
  size_t avgBlockQ=0;
  size_t avgBlockX=0;
  size_t maxBlockX=0;
  size_t maxBlockQ=0;


  for(i=0;i<cM.c;i++){
    maxTemp=0;
    for(j=0;j<cM.r;j++){
      //printf("%d ",cM.mat[IDX(i,j,cM.ld)]*PAD(groupCountQ[i])*PAD(groupCountX[j]));
      ans+=cM.mat[IDX(i,j,cM.ld)]*(groupCountQ[i])*(groupCountX[j]);
      avgBlocks+=cM.mat[IDX(i,j,cM.ld)];
      maxTemp+=cM.mat[IDX(i,j,cM.ld)]*PAD(groupCountX[j]);
    }
    //    printf("\n");
    if(maxBlocks < maxTemp){
      maxBlocks=maxTemp;
      maxBlocksInd=PAD(groupCountQ[i]);
    }
    //maxBlocks=MAX(maxTemp,maxBlocks);
  }
  
  for(i=0;i<cM.c;i++){
    avgBlockQ+=groupCountQ[i];
    avgBlockX+=groupCountX[i];
    maxBlockQ=MAX(maxBlockQ,groupCountQ[i]);
    maxBlockX=MAX(maxBlockX,groupCountX[i]);
  }
  
  printf("most amt of work for a query: %zu (%zu) ; avg = %6.4f \n",maxBlocks,maxBlocksInd,((double)ans)/((double)cM.c));
  printf("avg blocks/query block = %6.4f ; \n",((double)avgBlocks)/((double)cM.c));
  printf("avg blockQ = %6.4f; max = %zu \n",((double)avgBlockQ)/((double)cM.c),maxBlockQ);
  printf("avg blockX = %6.4f; max = %zu \n",((double)avgBlockX)/((double)cM.c),maxBlockX);
  
  return ans;
}


void kMinsWrap(matrix dD, matrix dMins, intMatrix dNNs){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(1,dD.pr/BLOCK_SIZE);
  
  kMinsKernel<<<grid,block>>>(dD,dMins,dNNs);
  cudaThreadSynchronize();
}


__global__ void kMinsKernel(matrix,matrix,intMatrix);

__global__ void kMinsKernel(matrix D, matrix dMins, intMatrix NNs){
  
  int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int ro = threadIdx.y; //row offset
  int co = threadIdx.x; //col offset

  __shared__ real smin[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int smp[BLOCK_SIZE][BLOCK_SIZE];

  real min, t;
  int mp; //min position
  int i,j,c;
  
   
  for(i=0 ; i<NNs.c ; i++){
    
    min = MAX_REAL;
    for(j=0 ; j<D.pc/BLOCK_SIZE ; j++){
      c = j*BLOCK_SIZE;
      if( c+co < D.c ){
	t = D.mat[ IDX( row, c+co, D.ld ) ];
	
	if( t < min ){
	  min = t;
	  mp = c+co;
	}
      }
    }
    
    smin[ro][co] = min;
    smp[ro][co] = mp;
    __syncthreads();


    for(j=BLOCK_SIZE/2 ; j>0 ; j/=2){
      if( co < j ){
	if( smin[ro][co+j] < smin[ro][co] ){
	  smin[ro][co] = smin[ro][co+j];
	  smp[ro][co] = smp[ro][co+j];
	}
      }
      __syncthreads();
    }

    if(co==0){
      NNs.mat[ IDX( row, i, NNs.ld ) ] = smp[ro][0];
      dMins.mat[ IDX( row, i, dMins.ld ) ] = smin[ro][0];

      D.mat[ IDX( row, smp[ro][0], D.ld ) ] = MAX_REAL;
    }

    __syncthreads();
  }
}

void brute2Step(matrix,matrix,intMatrix);


void brute2Step(matrix x, matrix q, intMatrix NNs){

  matrix dx, dq, dD, dMins;
  intMatrix dNNs;
  real *ranges, *dranges;
  
  copyAndMove(&dx, &x);
  copyAndMove(&dq, &q);
  copyAndMoveI(&dNNs, &NNs); //NNs.mat is garbage, but no matter.
  
  dD.r=q.r; dD.pr=q.pr; dD.c=x.r; dD.pc=x.pr; dD.ld=dD.pc;
  cudaMalloc( (void**)&dD.mat, dD.pr*dD.pc*sizeof(*dD.mat) );
  
  dMins.r=NNs.r; dMins.pr=NNs.pr; dMins.c=NNs.c; dMins.pc=NNs.pc; dMins.ld=NNs.ld;
  cudaMalloc( (void**)&dMins.mat, dMins.pr*dMins.pc*sizeof(*dMins.mat) );
  
  ranges = (real*)calloc( q.pr, sizeof(*ranges) );
  cudaMalloc( (void**)&dranges, q.pr*sizeof(*dranges) );

  dist1Wrap(dq,dx,dD);

  kMinsWrap(dD, dMins, dNNs);
  cudaMemcpy(NNs.mat, dNNs.mat, NNs.pr*NNs.pc*sizeof(*NNs.mat), cudaMemcpyDeviceToHost);
 
  free(ranges);
  cudaFree(dranges);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
  cudaFree(dD.mat);
  cudaFree(dNNs.mat);
  cudaFree(dMins.mat);

}


memPlan createMemPlan(int,int);

memPlan createMemPlan(unint nPts, unint memPerPt){
  memPlan mp;
  unsigned int memFree, memTot;
  unint ptsAtOnce;

  cuMemGetInfo(&memFree, &memTot);
  memFree = (unint)(((float)memFree)*MEM_USABLE);
  printf("memfree = %d \n",memFree);
  ptsAtOnce = DPAD(memFree/memPerPt); //max number of pts that can be processed at once
  printf("ptsAtOnce = %d \n",ptsAtOnce);
  mp.numComputeSegs = nPts/ptsAtOnce + ((nPts%ptsAtOnce==0) ? 0 : 1);
  mp.normSegSize=PAD(nPts/mp.numComputeSegs); 
  mp.lastSegSize=PAD(nPts) - mp.normSegSize*(mp.numComputeSegs-1);
  //Note that lastSegSize is automatically padded if nPts is.
  return mp;
}

typedef struct {
  unint numComputeSegs;
  unint normSegSize;//The number of points handled in one computation,
                     //though there will always be one leftover segment
                     //with (possibly) a different number of points.
  unint lastSegSize;//.. and this is it.
} memPlan;


void blockIntersection(charMatrix,matrix,real*,real*);

void blockIntersection(charMatrix cM, matrix dr, real *radiiX, real *radiiQ){
  matrix dD;
  real *dradiiX, *dradiiQ;
  unint pnR = dr.pr;
  charMatrix dcM;
  
  dD.r=dD.c=dr.r; dD.pr=dD.pc=dD.ld=dr.pr;
  dcM.r=cM.r; dcM.c=cM.c; dcM.pr=cM.pr; dcM.pc=cM.pc; dcM.ld=cM.ld;
  
  checkErr( cudaMalloc((void**)&dD.mat, pnR*pnR*sizeof(*dD.mat)) );
  checkErr( cudaMalloc((void**)&dradiiX, pnR*sizeof(*dradiiX)) );
  checkErr( cudaMalloc((void**)&dradiiQ, pnR*sizeof(*dradiiQ)) );
  checkErr( cudaMalloc((void**)&dcM.mat, dcM.pr*dcM.pc*sizeof(*dcM.mat)) );
  
  // Copying over the radii. Note that everything after the first dr.r places 
  // on the device variables is undefined.
  cudaMemcpy(dradiiX,radiiX,dr.r*sizeof(*dradiiX),cudaMemcpyHostToDevice);
  cudaMemcpy(dradiiQ,radiiQ,dr.r*sizeof(*dradiiQ),cudaMemcpyHostToDevice);
  
  dist1Wrap(dr, dr, dD);
  pruneWrap(dcM, dD, dradiiX, dradiiQ);

  cudaMemcpy(cM.mat,dcM.mat,pnR*pnR*sizeof(*dcM.mat),cudaMemcpyDeviceToHost);
  
  cudaFree(dcM.mat);
  cudaFree(dradiiQ);
  cudaFree(dradiiX);
  cudaFree(dD.mat);
}



void groupPoints(matrix,unint*,unint*,unint);

// This function sorts points by their repID.  It makes two passes through the 
// matrix x; one to count the bucket sizes, the next to place points in the 
// correct bucket.  Note that this function allocates a temporary
// matrix the size of x, then copies the results over to x at the end.  The 
// sort could be done in place, eg by doing numReps passes through x instead of 2.
void groupPoints(matrix x, unint *xID, unint *repIDs, unint numReps){
  matrix y;
  unint n=x.r;
  unint d=x.c;
  unint i;
  unint *gS; //groupSize
  unint *yID;

  yID = (unint*)calloc(n,sizeof(*yID));
  y.mat = (real*)calloc(n*d,sizeof(*y.mat));
  gS = (unint*)calloc(numReps+1,sizeof(*gS));

  y.r=n; y.pr=n; y.c=d; y.pc=d; y.ld=d;

  for(i=0;i<n;i++)
    gS[repIDs[i]+1]++;
  for(i=1;i<numReps;i++)
    gS[i]=gS[i-1]+gS[i];
  
  for(i=0;i<n;i++){
    copyVector(&y.mat[IDX(gS[repIDs[i]],0,y.ld)], &x.mat[IDX(i,0,x.ld)],d);
    yID[gS[repIDs[i]]]=xID[i];
    gS[repIDs[i]]++;
  }
  
  for(i=0;i<n;i++){
    copyVector(&x.mat[IDX(i,0,x.ld)],&y.mat[IDX(i,0,y.ld)],d);
    xID[i]=yID[i];
  }
  
  free(yID);
  free(gS);
  free(y.mat);
}

__global__ void pruneKernel(const matrix,const real*,const real*,charMatrix);


__global__ void pruneKernel(const matrix D, const real *radiiX, const real *radiiQ, charMatrix cM){
  unint offX = threadIdx.x;
  unint offQ = threadIdx.y;

  unint blockX = blockIdx.x * BLOCK_SIZE;
  unint blockQ = blockIdx.y * BLOCK_SIZE;
  
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
  }
}

void pruneWrap(charMatrix dcM, matrix dD, real *dradiiX, real *dradiiQ){
  dim3 block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid(dD.pr/BLOCK_SIZE,dD.pc/BLOCK_SIZE);
  
  pruneKernel<<<grid,block>>>(dD,dradiiX,dradiiQ,dcM);
  cudaThreadSynchronize();
}

void pruneWrap(charMatrix,matrix,real*,real*);
