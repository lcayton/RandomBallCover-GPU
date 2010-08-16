#ifndef RBC_CU
#define RBC_CU

#include<sys/time.h>
#include<stdio.h>
#include<cuda.h>
#include "utils.h"
#include "defs.h"
#include "utilsGPU.h"
#include "rbc.h"
#include "kernels.h"
#include "kernelWrap.h"
#include "sKernel.h"

void rbc(matrix x, matrix q, int numReps, int s, int *NNs){
  int i;
  matrix r; // random subset of x
  matrix dr; //device version of r
  int *repIDsQ; //assigment of each pt to a rep.
  real *radiiQ; //radius of each group
  int *groupCountX, *groupCountQ; //num pts in each group
  real *distToRepsQ; //distance of each pt to its rep
  int *qID;
  charMatrix cM; //compute matrix.  will be treated as a binary matrix
  intMatrix xmap;
  compPlan cP; 
  int compLength;
  struct timeval tvB, tvE;
      
  //convenience variables
  int n = x.r; //num pts in DB
  int m = q.r; //num queries
  int pnr = PAD(numReps);
  
  r.r=numReps; r.c=x.c; r.pr=pnr; r.pc=PAD(r.c); r.ld=r.pc;
  r.mat = (real*)calloc(r.pr*r.pc,sizeof(*(r.mat)));

  repIDsQ = (int*)calloc(m,sizeof(*repIDsQ));
  radiiQ = (real*)calloc(pnr,sizeof(*radiiQ));
  groupCountX = (int*)calloc(pnr,sizeof(*groupCountX));
  groupCountQ = (int*)calloc(pnr,sizeof(*groupCountQ));
  distToRepsQ = (real*)calloc(m,sizeof(*distToRepsQ));

  qID = (int*)calloc(PAD(m+(BLOCK_SIZE-1)*pnr),sizeof(*qID));

  cM.mat = (char*)calloc(pnr*pnr,sizeof(*cM.mat));
  cM.r=numReps; cM.c=numReps; cM.pr=pnr; cM.pc=pnr; cM.ld=cM.pc;

  xmap.r=numReps; xmap.pr=PAD(numReps); xmap.c=s; xmap.pc=PAD(s); xmap.ld=xmap.pc;
  cudaMalloc( (void**)&xmap.mat, xmap.pr*xmap.pc*sizeof(*xmap.mat) );

  //The following lines of code init xmap to 0s.
  intMatrix tempXmap;
  tempXmap.r=xmap.r; tempXmap.pr=xmap.pr; tempXmap.c=xmap.c; tempXmap.pc=xmap.pc; tempXmap.ld=xmap.ld;
  tempXmap.mat = (int*)calloc( tempXmap.pr*tempXmap.pc, sizeof(*tempXmap.mat) );
  cudaMemcpy(xmap.mat, tempXmap.mat, xmap.pr*xmap.pc*sizeof(*xmap.mat), cudaMemcpyHostToDevice);
  free(tempXmap.mat);

  
  //Choose representatives and move them to device
  int *randInds;
  randInds = (int*)calloc(pnr,sizeof(*randInds));
  subRandPerm(numReps,n,randInds);
  
  for(i=0;i<numReps;i++){
    copyVector(&r.mat[IDX(i,0,r.ld)], &x.mat[IDX(randInds[i],0,x.ld)], x.c);
  }
  free(randInds);

  dr.r=r.r; dr.c=r.c; dr.pr=r.pr; dr.pc=r.pc; dr.ld=r.ld;
  cudaMalloc( (void**)&(dr.mat), dr.pr*dr.pc*sizeof(*(dr.mat)) );
  cudaMemcpy(dr.mat,r.mat,dr.pr*dr.pc*sizeof(*(dr.mat)),cudaMemcpyHostToDevice);

  
  matrix dx;
  copyAndMove(&dx, &x);
  
  build(dx, dr, xmap, groupCountX, s); 
  
  /* tempXmap.r=xmap.r; tempXmap.pr=xmap.pr; tempXmap.c=xmap.c; tempXmap.pc=xmap.pc; tempXmap.ld=xmap.ld; */
  /* tempXmap.mat = (int*)calloc( tempXmap.pr*tempXmap.pc, sizeof(*tempXmap.mat) ); */
  /* cudaMemcpy(tempXmap.mat, xmap.mat, xmap.pr*xmap.pc*sizeof(*xmap.mat), cudaMemcpyDeviceToHost); */
  /* for( i=0; i<16; i++ ) */
  /*   printf("%d ",tempXmap.mat[i]); */
  /* printf("\n"); */
  /* free(tempXmap.mat); */


  gettimeofday(&tvB,NULL);  //Start the timer for the queries
  
  matrix dqOrig;
  copyAndMove(&dqOrig, &q);

  computeReps(dqOrig, dr, repIDsQ, distToRepsQ);

  //How many points are assigned to each group?
  computeCounts(repIDsQ, m, groupCountQ);
  
  
  //Set up the mapping from groups to queries (qID).
  buildQMap(q, qID, repIDsQ, numReps, &compLength);

  // Determine which blocks need to be compared with one another and
  // store the results in the computation matrix cM.
  //blockIntersection(cM, dr, radiiX, radiiQ);
  idIntersection(cM);

  // Setup the compute plan
  initCompPlan(&cP, cM, groupCountQ, groupCountX, numReps);

  // Compute the NNs according to the compute plan
  computeNNs(dx, dqOrig, xmap, cP, qID, NNs, compLength);
  
  gettimeofday(&tvE,NULL);
  printf("\t.. time elapsed (for queries) = %6.4f \n",timeDiff(tvB,tvE));

  cudaFree(dr.mat); 
  cudaFree(dqOrig.mat);
  freeCompPlan(&cP);
  free(cM.mat);
  free(distToRepsQ);
  free(groupCountX);
  free(groupCountQ);
  free(qID);
  free(radiiQ);
  free(repIDsQ);
  free(r.mat);
  cudaFree(xmap.mat);
  cudaFree(dx.mat);
}


void build(const matrix dx, const matrix dr, intMatrix xmap, int *counts, int s){
  
  int n = dx.pr;
  int p = dr.r;

  //Figure out how much fits into memory
  unsigned int memFree, memTot;
  cuMemGetInfo(&memFree, &memTot);
  memFree = (int)(((float)memFree)*MEM_USABLE);
     
  //mem needed per rep = n*sizeof(real)+n*sizeof(char)+n*sizeof(int)+sizeof(real)+sizeof(int)
  //                   = dist mat      +ir            +dSums        +range       +dCnts
  int ptsAtOnce = DPAD(memFree/((n+1)*sizeof(real) + n*sizeof(char) +(n+1)*sizeof(int)));
  if(ptsAtOnce==0){
    fprintf(stderr,"memfree = %d \n",memFree);
    fprintf(stderr,"error: not enough memory to build the RBC.. exiting\n");
    exit(1);
  }

  matrix dD;
  dD.pr=dD.r=ptsAtOnce; dD.c=dx.r; dD.pc=dx.pr; dD.ld=dD.pc;
  cudaMalloc( (void**)&dD.mat, dD.pr*dD.pc*sizeof(*dD.mat) );

  real *dranges;
  cudaMalloc( (void**)&dranges, ptsAtOnce*sizeof(real) );

  charMatrix ir;
  ir.r=dD.r; ir.pr=dD.pr; ir.c=dD.c; ir.pc=dD.pc; ir.ld=dD.ld;
  ir.mat = (char*)calloc( ir.pr*ir.pc, sizeof(*ir.mat) );
  charMatrix dir;
  copyAndMoveC(&dir, &ir);

  intMatrix dSums; //used to compute memory addresses.
  dSums.r=dir.r; dSums.pr=dir.pr; dSums.c=dir.c; dSums.pc=dir.pc; dSums.ld=dir.ld;
  cudaMalloc( (void**)&dSums.mat, dSums.pc*dSums.pr*sizeof(*dSums.mat) );

  int *dCnts;
  cudaMalloc( (void**)&dCnts, ptsAtOnce*sizeof(*dCnts) );
    
   
  int numits=0;
  int numLeft = p; //points left to process
  int row = 0; //base row for iteration of while loop
  int pi, pip; //pi=pts per it, pip=pad(pi)
  
  while( numLeft > 0 ){
    numits++;
    pi = MIN(ptsAtOnce, numLeft);  //points to do this iteration.
    pip = PAD(pi);
    dD.r = pi; dD.pr = pip; dir.r=pi; dir.pr=pip; dSums.r=pi; dSums.pr=pip;
  
    distSubMat(dr, dx, dD, row, pip); //compute the distance matrix
    findRangeWrap(dD, dranges, s);  //find an appropriate range
    rangeSearchWrap(dD, dranges, dir); //set binary vector for points in range
    
    sumWrap(dir, dSums);  //This and the next call perform the parallel compaction.

    buildMapWrap(xmap, dir, dSums, row);
    getCountsWrap(dCnts,dir,dSums);  //How many points are assigned to each rep?  It is not 
                                     //*exactly* s, which is why we need to compute this.
    cudaMemcpy( &counts[row], dCnts, pi*sizeof(*counts), cudaMemcpyDeviceToHost );
    
    numLeft -= pi;
    row += pi;
  }
 
  cudaFree(dCnts);
  free(ir.mat);
  cudaFree(dranges);
  cudaFree(dir.mat);
  cudaFree(dSums.mat);
  cudaFree(dD.mat);
}


//Assign each point in dq to its nearest point in dr.  
void computeReps(matrix dq, matrix dr, int *repIDs, real *distToReps){
  real *dMins;
  int *dMinIDs;

  cudaMalloc((void**)&(dMins), dq.pr*sizeof(*dMins));
  cudaMalloc((void**)&(dMinIDs), dq.pr*sizeof(*dMinIDs));
  
  nnWrap(dq,dr,dMins,dMinIDs);
  
  cudaMemcpy(distToReps,dMins,dq.r*sizeof(*dMins),cudaMemcpyDeviceToHost);
  cudaMemcpy(repIDs,dMinIDs,dq.r*sizeof(*dMinIDs),cudaMemcpyDeviceToHost);
  
  cudaFree(dMins);
  cudaFree(dMinIDs);
}



//Assumes radii is initialized to 0s
void computeRadii(int *repIDs, real *distToReps, real *radii, int n, int numReps){
  int i;

  for(i=0;i<n;i++)
    radii[repIDs[i]] = MAX(distToReps[i],radii[repIDs[i]]);
}


//Assumes groupCount is initialized to 0s
void computeCounts(int *repIDs, int n, int *groupCount){
  int i;
  
  for(i=0;i<n;i++)
    groupCount[repIDs[i]]++;
}


// This function computes a cumulative sum of the groupCounts.
// Assumes groupOff is initialized to 0s.  
void computeOffsets(int *groupCount, int n, int *groupOff){
  int i;

  for(i=1;i<n;i++)
    groupOff[i]=groupOff[i-1]+PAD(groupCount[i-1]);
}


// This function sorts points by their repID.  It makes two passes through the 
// matrix x; one to count the bucket sizes, the next to place points in the 
// correct bucket.  Note that this function unfortunately allocates a temporary
// matrix the size of x, then copies the results over to x at the end.  The 
// sort could be done in place, eg by doing numReps passes through x instead of 2.
void groupPoints(matrix x, int *xID, int *repIDs, int numReps){
  matrix y;
  int n=x.r;
  int d=x.c;
  int i;
  int *gS; //groupSize
  int *yID;

  yID = (int*)calloc(n,sizeof(*yID));
  y.mat = (real*)calloc(n*d,sizeof(*y.mat));
  gS = (int*)calloc(numReps+1,sizeof(*gS));

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


void buildQMap(matrix q, int *qID, int *repIDs, int numReps, int *compLength){
  int n=q.r;
  int i;
  int *gS; //groupSize
  
  gS = (int*)calloc(numReps+1,sizeof(*gS));
  
  for( i=0; i<n; i++ )
    gS[repIDs[i]+1]++;
  for( i=0; i<numReps+1; i++ )
    gS[i]=PAD(gS[i]);
  
  for( i=1; i<numReps+1; i++ )
    gS[i]=gS[i-1]+gS[i];
  
  *compLength = gS[numReps];
  
  for( i=0; i<(*compLength); i++ )
    qID[i]=DUMMY_IDX;
  
  for( i=0; i<n; i++ ){
    qID[gS[repIDs[i]]]=i;
    gS[repIDs[i]]++;
  }

  free(gS);
}


void blockIntersection(charMatrix cM, matrix dr, real *radiiX, real *radiiQ){
  matrix dD;
  real *dradiiX, *dradiiQ;
  int pnR = dr.pr;
  charMatrix dcM;
  
  dD.r=dD.c=dr.r; dD.pr=dD.pc=dD.ld=dr.pr;
  dcM.r=cM.r; dcM.c=cM.c; dcM.pr=cM.pr; dcM.pc=cM.pc; dcM.ld=cM.ld;
  
  cudaMalloc((void**)&dD.mat, pnR*pnR*sizeof(*dD.mat));
  cudaMalloc((void**)&dradiiX, pnR*sizeof(*dradiiX));
  cudaMalloc((void**)&dradiiQ, pnR*sizeof(*dradiiQ));
  cudaMalloc((void**)&dcM.mat, dcM.pr*dcM.pc*sizeof(*dcM.mat));
  
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


// Sets the computation matrix to the identity.  
void idIntersection(charMatrix cM){
  int i;
  for(i=0;i<cM.r;i++){
    if(i<cM.c)
      cM.mat[IDX(i,i,cM.ld)]=1;
  }
}



void fullIntersection(charMatrix cM){
  int i,j;
  for(i=0;i<cM.r;i++){
    for(j=0;j<cM.c;j++){
      cM.mat[IDX(i,j,cM.ld)]=1;
    }
  }
}


// Danger: this function allocates memory that it does not free.  
// Use freeCompPlan to clear mem.
void initCompPlan(compPlan *cP, charMatrix cM, int *groupCountQ, int *groupCountX, int numReps){
  int i,j,k;
  int maxNumGroups=0;
  int *groupOff;
  
  groupOff = (int*)calloc(numReps,sizeof(*groupOff));

  cP->sNumGroups=numReps;
  cP->numGroups = (int*)calloc(cP->sNumGroups,sizeof(*cP->numGroups));
  
  for(i=0;i<numReps;i++){
    cP->numGroups[i] = 0;
    for(j=0;j<numReps;j++){
      cP->numGroups[i] += cM.mat[IDX(i,j,cM.ld)];
    }
    maxNumGroups = MAX(cP->numGroups[i],maxNumGroups);
  }
  
  cP->ld = maxNumGroups;

  cP->sGroupCountX = maxNumGroups*numReps;
  cP->groupCountX = (int*)calloc(cP->sGroupCountX,sizeof(*cP->groupCountX));
  cP->sGroupOff = maxNumGroups*numReps;
  cP->groupOff = (int*)calloc(cP->sGroupOff,sizeof(*cP->groupOff));
  
  computeOffsets(groupCountX,numReps,groupOff);
  
  int tempSize=0;
  for(i=0;i<numReps;i++)
    tempSize+=PAD(groupCountQ[i]);
  
  cP->sQToGroup = tempSize;
  cP->qToGroup = (int*)calloc(cP->sQToGroup,sizeof(*cP->qToGroup));
  
  for(i=0, k=0;i<numReps;i++){
    for(j=0;j<PAD(groupCountQ[i]);j++){
      cP->qToGroup[k]=i;
      k++;
    }
  }
  
  for(i=0;i<numReps;i++){
    for(j=0, k=0;j<numReps;j++){
      if(cM.mat[IDX(i,j,cM.ld)]){
  	cP->groupCountX[IDX(i,k,cP->ld)]=groupCountX[j];
  	cP->groupOff[IDX(i,k,cP->ld)]=groupOff[j];
  	k++;
      }
    }
  }
  free(groupOff);
}


//Frees memory allocated in initCompPlan.
void freeCompPlan(compPlan *cP){
  free(cP->numGroups);
  free(cP->groupCountX);
  free(cP->groupOff);
  free(cP->qToGroup);
}


void computeNNs(matrix dx, matrix dq, intMatrix xmap, compPlan cP, int *qIDs, int *NNs, int compLength){
  compPlan dcP;
  real *dMins;
  int *dMinIDs;

  dcP.ld=cP.ld;
  cudaMalloc((void**)&dcP.numGroups,cP.sNumGroups*sizeof(*dcP.numGroups));
  cudaMalloc((void**)&dcP.groupCountX,cP.sGroupCountX*sizeof(*dcP.groupCountX));
  cudaMalloc((void**)&dcP.groupOff,cP.sGroupOff*sizeof(*dcP.groupOff));
  cudaMalloc((void**)&dcP.qToGroup,cP.sQToGroup*sizeof(*dcP.qToGroup));

  cudaMemcpy(dcP.numGroups,cP.numGroups,cP.sNumGroups*sizeof(*dcP.numGroups),cudaMemcpyHostToDevice);
  cudaMemcpy(dcP.groupCountX,cP.groupCountX,cP.sGroupCountX*sizeof(*dcP.groupCountX),cudaMemcpyHostToDevice);
  cudaMemcpy(dcP.groupOff,cP.groupOff,cP.sGroupOff*sizeof(*dcP.groupOff),cudaMemcpyHostToDevice);
  cudaMemcpy(dcP.qToGroup,cP.qToGroup,cP.sQToGroup*sizeof(*dcP.qToGroup),cudaMemcpyHostToDevice);
  
  cudaMalloc((void**)&dMins,compLength*sizeof(*dMins));
  cudaMalloc((void**)&dMinIDs,compLength*sizeof(*dMinIDs));

  int *dqIDs;
  cudaMalloc( (void**)&dqIDs, compLength*sizeof(*dqIDs) );
  cudaMemcpy( dqIDs, qIDs, compLength*sizeof(*dqIDs), cudaMemcpyHostToDevice );
  

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid;
  dimGrid.x = 1;
  int numDone = 0;
  while( numDone<compLength ){
    int todo = MIN( (compLength-numDone) , MAX_BS*BLOCK_SIZE );

    dimGrid.y = todo/BLOCK_SIZE;
    planNNKernel<<<dimGrid,dimBlock>>>(dq,dx,dMins,dMinIDs,dcP,xmap,dqIDs,numDone);
    cudaThreadSynchronize();
    numDone += todo;
  }

  cudaMemcpy( NNs, dMinIDs, dq.r*sizeof(*NNs), cudaMemcpyDeviceToHost);
  
  cudaFree(dcP.numGroups);
  cudaFree(dcP.groupCountX);
  cudaFree(dcP.groupOff);
  cudaFree(dcP.qToGroup);
  cudaFree(dMins);
  cudaFree(dMinIDs);
  cudaFree(dqIDs);
}


//This calls the dist1Kernel wrapper, but has it compute only 
//a submatrix of the all-pairs distance matrix.  In particular,
//only distances from dr[start,:].. dr[start+length-1] to all of x
//are computed, resulting in a distance matrix of size 
//length by dx.pr.  It is assumed that length is padded.
void distSubMat(matrix dr, matrix dx, matrix dD, int start, int length){
  dr.r=dr.pr=length;
  dr.mat = &dr.mat[IDX( start, 0, dr.ld )];
  dist1Wrap(dr, dx, dD);
}


#endif
