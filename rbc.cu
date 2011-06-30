/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

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
#include "sKernelWrap.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

void queryRBC(const matrix q, const rbcStruct rbcS, unint *NNs, real* NNdists){
  unint m = q.r;
  unint numReps = rbcS.dr.r;
  unint compLength;
  compPlan dcP;
  unint *qMap, *dqMap;
  qMap = (unint*)calloc(PAD(m+(BLOCK_SIZE-1)*PAD(numReps)),sizeof(*qMap));
  matrix dq;
  copyAndMove(&dq, &q);
  
  charMatrix cM;
  cM.r=cM.c=numReps; cM.pr=cM.pc=cM.ld=PAD(numReps);
  cM.mat = (char*)calloc( cM.pr*cM.pc, sizeof(*cM.mat) );
  
  unint *repIDsQ;
  repIDsQ = (unint*)calloc( m, sizeof(*repIDsQ) );
  real *distToRepsQ;
  distToRepsQ = (real*)calloc( m, sizeof(*distToRepsQ) );
  unint *groupCountQ;
  groupCountQ = (unint*)calloc( PAD(numReps), sizeof(*groupCountQ) );
  
  computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

  //How many points are assigned to each group?
  computeCounts(repIDsQ, m, groupCountQ);
  
  //Set up the mapping from groups to queries (qMap).
  buildQMap(q, qMap, repIDsQ, numReps, &compLength);
  
  // Setup the computation matrix.  Currently, the computation matrix is 
  // just the identity matrix: each query assigned to a particular 
  // representative is compared only to that representative's points.  
  idIntersection(cM);

  initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

  checkErr( cudaMalloc( (void**)&dqMap, compLength*sizeof(*dqMap) ) );
  cudaMemcpy( dqMap, qMap, compLength*sizeof(*dqMap), cudaMemcpyHostToDevice );
  
  computeNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);
  
  free(qMap);
  cudaFree(dqMap);
  freeCompPlan(&dcP);
  cudaFree(dq.mat);
  free(cM.mat);
  free(repIDsQ);
  free(distToRepsQ);
  free(groupCountQ);
}

//This function is very similar to queryRBC, with a couple of basic changes to handle
//k-nn.  
void kqueryRBC(const matrix q, const rbcStruct rbcS, intMatrix NNs, matrix NNdists){
  unint m = q.r;
  unint numReps = rbcS.dr.r;
  unint compLength;
  compPlan dcP;
  unint *qMap, *dqMap;
  qMap = (unint*)calloc(PAD(m+(BLOCK_SIZE-1)*PAD(numReps)),sizeof(*qMap));
  matrix dq;
  copyAndMove(&dq, &q);
  
  charMatrix cM;
  cM.r=cM.c=numReps; cM.pr=cM.pc=cM.ld=PAD(numReps);
  cM.mat = (char*)calloc( cM.pr*cM.pc, sizeof(*cM.mat) );
  
  unint *repIDsQ;
  repIDsQ = (unint*)calloc( m, sizeof(*repIDsQ) );
  real *distToRepsQ;
  distToRepsQ = (real*)calloc( m, sizeof(*distToRepsQ) );
  unint *groupCountQ;
  groupCountQ = (unint*)calloc( PAD(numReps), sizeof(*groupCountQ) );
  
  computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

  //How many points are assigned to each group?
  computeCounts(repIDsQ, m, groupCountQ);
  
  //Set up the mapping from groups to queries (qMap).
  buildQMap(q, qMap, repIDsQ, numReps, &compLength);
  
  // Setup the computation matrix.  Currently, the computation matrix is 
  // just the identity matrix: each query assigned to a particular 
  // representative is compared only to that representative's points.  

  // NOTE: currently, idIntersection is the *only* computation matrix 
  // that will work properly with k-nn search (this is not true for 1-nn above).
  idIntersection(cM);

  initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

  checkErr( cudaMalloc( (void**)&dqMap, compLength*sizeof(*dqMap) ) );
  cudaMemcpy( dqMap, qMap, compLength*sizeof(*dqMap), cudaMemcpyHostToDevice );
  
  computeKNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);

  free(qMap);
  freeCompPlan(&dcP);
  cudaFree(dq.mat);
  free(cM.mat);
  free(repIDsQ);
  free(distToRepsQ);
  free(groupCountQ);
}

void buildVor( const matrix x, vorStruct *vorS, unint numReps, unint ol){
  ol = MIN( ol, 32 );
  printf(" ol = %d \n", ol);
  unint i, j;
  unint n = x.pr;
  
  setupRepsVor( x, vorS, numReps );

  matrix dists;
  initMat( &dists, x.r, KMAX );
  dists.mat = (real*)calloc( sizeOfMat(dists), sizeof(dists.mat) );
  intMatrix nns; 
  initIntMat( &nns, x.r, KMAX );
  nns.mat = (unint*)calloc( sizeOfIntMat(nns), sizeof(nns.mat) );

  matrix zeros;
  initMat( &zeros, BLOCK_SIZE, x.c );
  zeros.mat = (real*)calloc( sizeOfMat(zeros), sizeof(*zeros.mat) );

  matrix dr;
  initMat( &dr, numReps, x.c );
  checkErr( cudaMalloc( (void**)&dr.mat, sizeOfMatB( dr ) ) );
  cudaMemcpy( dr.mat, vorS->r.mat, sizeOfMatB(dr), cudaMemcpyHostToDevice );

  //assume all of x is in CPU RAM
  //see how much fits onto the GPU at once.
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (size_t)(((double)memFree)*MEM_USABLE);
  unint memPerPt = x.pc*sizeof(*x.mat) + sizeof(real) + sizeof(unint);
  unint ptsAtOnce = MIN( DPAD(memFree/memPerPt), n );

  matrix dx;
  initMat( &dx, ptsAtOnce, x.c );
  checkErr( cudaMalloc( (void**)&dx.mat, sizeOfMatB( dx ) ) );
  matrix ddists;
  initMat( &ddists, ptsAtOnce, KMAX );
  checkErr( cudaMalloc( (void**)&ddists.mat, sizeOfMatB( ddists ) ) );
  intMatrix dnns;
  initIntMat( &dnns, ptsAtOnce, KMAX );
  checkErr( cudaMalloc( (void**)&dnns.mat, sizeOfIntMatB( dnns ) ) );

  // loop through the X, finding the NN for each.
  unint numLeft = n;
  unint row = 0;
  unint pi, pip;
  unint its = 0;
  while( numLeft > 0 ){
    pi = MIN( ptsAtOnce, numLeft );
    pip = PAD( pi );
    cudaMemcpy( dx.mat, &x.mat[IDX( row, 0, x.ld )], pi*x.pc*sizeof(*x.mat), cudaMemcpyHostToDevice );
    dx.r = pi; dx.pr = pip;

    //zero out extra rows of x.mat
    if( pip-pi )
      cudaMemcpy( &dx.mat[IDX( pi, 0, x.ld )], zeros.mat, (pip-pi)*x.pc*sizeof(*x.mat), cudaMemcpyHostToDevice );

    knnWrap( dx, dr, ddists, dnns );
    cudaMemcpy( &nns.mat[IDX( row, 0, nns.ld )], dnns.mat, pi*KMAX*sizeof(unint), cudaMemcpyDeviceToHost );

    numLeft -= pi;
    row += pi;
    printf("\t it %d finished; %d points left\n", its++, numLeft);
  }
  
  cudaFree( dx.mat );
  cudaFree( ddists.mat );
  cudaFree( dnns.mat );
  cudaFree( dr.mat );

  vorS->groupCount = (unint*)calloc( PAD(numReps), sizeof(*vorS->groupCount) );
  for( i=0; i<x.r; i++ )
    for( j=0; j<ol; j++ )
      vorS->groupCount[ nns.mat[IDX( i, j, nns.ld )] ]++;
  
  unint maxCount = 0;
  for( i=0; i<numReps; i++ )
    maxCount = MAX( maxCount, vorS->groupCount[i] );
  printf("max count is %d \n", maxCount );
  
  vorS->xMap = (unint**)calloc( numReps, sizeof(unint*) );
  for( i=0; i<numReps; i++ )
    vorS->xMap[i] = (unint*)calloc( vorS->groupCount[i], sizeof(unint) );
  
  unint *pos = (unint*)calloc( numReps, sizeof(*pos) );
  
  for( i=0; i<x.r; i++ ){
    for( j=0; j<ol; j++ ){
      unint ind = nns.mat[IDX( i, j, nns.ld )];
      vorS->xMap[ind][pos[ind]++] = i;
    }
  }
  
  free( pos );
  free( nns.mat );
  free( dists.mat );
  free( zeros.mat );
}
  

//out-of-memory version of buildVor
void buildVorBig( const hdMatrix x, vorStruct *vorS, unint numReps, unint ol){
  ol = MIN( ol, 32 );
  printf(" ol = %d \n", ol);
  unint i, j;
  unint n = x.r; 
  
  setupRepsVorHD( x, vorS, numReps );

  matrix dists;
  initMat( &dists, x.r, KMAX );
  dists.mat = (real*)calloc( sizeOfMat(dists), sizeof(dists.mat) );
  intMatrix nns; 
  initIntMat( &nns, x.r, KMAX );
  nns.mat = (unint*)calloc( sizeOfIntMat(nns), sizeof(nns.mat) );

  matrix zeros;
  initMat( &zeros, BLOCK_SIZE, x.c );
  zeros.mat = (real*)calloc( sizeOfMat(zeros), sizeof(*zeros.mat) );

  matrix dr;
  initMat( &dr, numReps, x.c );
  checkErr( cudaMalloc( (void**)&dr.mat, sizeOfMatB( dr ) ) );
  cudaMemcpy( dr.mat, vorS->r.mat, sizeOfMatB(dr), cudaMemcpyHostToDevice );

  //assume all of x is in CPU RAM
  //see how much fits onto the GPU at once.
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (size_t)(((double)memFree)*MEM_USABLE);
  unint memPerPt = PAD(x.c)*sizeof(real) + KMAX*sizeof(real) + KMAX*sizeof(unint);
  unint ptsAtOnce = MIN( DPAD(memFree/memPerPt), n );
  printf("ptsAtOnce = %d \n", ptsAtOnce);

  //for now, load from hd to mem to gpu mem.
  matrix xmem;
  initMat( &xmem, ptsAtOnce, x.c );
  xmem.mat = (real*)calloc( sizeOfMat(xmem), sizeof(*xmem.mat) );

  matrix dx;
  initMat( &dx, ptsAtOnce, x.c );
  checkErr( cudaMalloc( (void**)&dx.mat, sizeOfMatB( dx ) ) );
  matrix ddists;
  initMat( &ddists, ptsAtOnce, KMAX );
  checkErr( cudaMalloc( (void**)&ddists.mat, sizeOfMatB( ddists ) ) );
  intMatrix dnns;
  initIntMat( &dnns, ptsAtOnce, KMAX );
  checkErr( cudaMalloc( (void**)&dnns.mat, sizeOfIntMatB( dnns ) ) );

  // loop through the X, finding the NN for each.
  unint numLeft = n;
  unint row = 0;
  unint pi, pip;
  unint its = 0;
  printf("starting loop...\n");
  while( numLeft > 0 ){
    pi = MIN( ptsAtOnce, numLeft );
    pip = PAD( pi );

    readBlock( xmem, 0, x, row, pi );
    cudaMemcpy( dx.mat, xmem.mat, pi*xmem.pc*sizeof(*xmem.mat), cudaMemcpyHostToDevice );
    dx.r = pi; dx.pr = pip;

    //zero out extra rows of x.mat
    if( pip-pi )
      cudaMemcpy( &dx.mat[IDX( pi, 0, dx.ld )], zeros.mat, (pip-pi)*dx.pc*sizeof(*zeros.mat), cudaMemcpyHostToDevice );

    knnWrap( dx, dr, ddists, dnns );
    cudaMemcpy( &nns.mat[IDX( row, 0, nns.ld )], dnns.mat, pi*KMAX*sizeof(unint), cudaMemcpyDeviceToHost );

    numLeft -= pi;
    row += pi;
    printf("\t it %d finished; %d points left\n", its++, numLeft);
  }
  
  cudaFree( dx.mat );
  cudaFree( ddists.mat );
  cudaFree( dnns.mat );
  cudaFree( dr.mat );

  vorS->groupCount = (unint*)calloc( PAD(numReps), sizeof(*vorS->groupCount) );
  for( i=0; i<x.r; i++ )
    for( j=0; j<ol; j++ )
      vorS->groupCount[ nns.mat[IDX( i, j, nns.ld )] ]++;
  
  unint maxCount = 0;
  for( i=0; i<numReps; i++ )
    maxCount = MAX( maxCount, vorS->groupCount[i] );
  printf("max count is %d \n", maxCount );
  
  vorS->xMap = (unint**)calloc( numReps, sizeof(unint*) );
  for( i=0; i<numReps; i++ )
    vorS->xMap[i] = (unint*)calloc( vorS->groupCount[i], sizeof(unint) );
  
  unint *pos = (unint*)calloc( numReps, sizeof(*pos) );
  
  for( i=0; i<x.r; i++ ){
    for( j=0; j<ol; j++ ){
      unint ind = nns.mat[IDX( i, j, nns.ld )];
      vorS->xMap[ind][pos[ind]++] = i;
    }
  }

  free( xmem.mat );
  free( pos );
  free( nns.mat );
  free( dists.mat );
  free( zeros.mat );
}


#define RPI 1024 //reps per it
//s needs to be a multiple of 32, numReps a multiple of RPI
void buildBigRBC(const matrix x, rbcStruct *rbcS, unint numReps, unint s){
  unint i;
  unint n = x.pr; 

  setupReps( x, rbcS, numReps );
  intMatrix xmap; 
  initIntMat( &xmap, numReps, s ); 
  xmap.mat = (unint*)calloc( sizeOfIntMat(xmap), sizeof(*xmap.mat) );

  intMatrix dhi; //a subset of xMap (used as heap)
  matrix dh; //corresponding dists
  initIntMat( &dhi, RPI, s );
  initMat( &dh, RPI, s );
  cudaMalloc( (void**)&dhi.mat, sizeOfIntMatB( dhi ) );
  cudaMalloc( (void**)&dh.mat, sizeOfMatB( dh ) );

  matrix dr; //subset of the reps
  initMat( &dr, RPI, x.c );
  cudaMalloc( (void**)&dr.mat, sizeOfMatB( dr ) );

  //dummy heap used for reseting
  matrix max_h;
  initMat( &max_h, RPI, s );
  max_h.mat = (real*)calloc( sizeOfMat(max_h), sizeof(*max_h.mat) );
  for( i=0; i<sizeOfMat(max_h); i++ )
    max_h.mat[i] = MAX_REAL;

  matrix zeros;
  initMat( &zeros, BLOCK_SIZE, x.c );
  zeros.mat = (real*)calloc( sizeOfMat(zeros), sizeof(*zeros.mat) );

  //assume all of x is in CPU RAM
  //see how much fits onto the GPU at once.
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (size_t)(((double)memFree)*MEM_USABLE);
  unint memPerPt = x.pc*sizeof(*x.mat);
  unint ptsAtOnce = MIN( DPAD(memFree/memPerPt), n );
  
  printf(" mem free for x = %6.2f, ptsAtOnce = %u \n", ((double)memFree)/(1024.0*1024.0), ptsAtOnce);

  //allocate the temp dx which is on the device.
  matrix dx;
  initMat( &dx, ptsAtOnce, x.c );
  checkErr( cudaMalloc( (void**)&dx.mat, sizeOfMatB( dx ) ) );

  for( i=0; i<numReps/RPI; i++ ){
    //reset the heap dists
    cudaMemcpy( dh.mat, max_h.mat, sizeOfMatB( dh ), cudaMemcpyHostToDevice );
    //copy the appropriate rows into dr
    cudaMemcpy( dr.mat, &rbcS->dr.mat[IDX( i*RPI, 0, rbcS->dr.ld )], sizeOfMatB( dr ), cudaMemcpyDeviceToDevice );
    
    unint numLeft = n;
    unint row = 0; //current row of x
    unint pi, pip;
    //unint its=0;
    while( numLeft > 0 ){
      pi = MIN( ptsAtOnce, numLeft );
      pip = PAD( pi );
      cudaMemcpy( dx.mat, &x.mat[IDX( row, 0, x.ld )], pi*x.pc*sizeof(*x.mat), cudaMemcpyHostToDevice );
      dx.r = pi; dx.pr = pip;

      //need to zero-out x.mat
      if( pip-pi )
	cudaMemcpy( &dx.mat[IDX( pi, 0, x.ld )], zeros.mat, (pip-pi)*x.pc*sizeof(*x.mat), cudaMemcpyHostToDevice );
      
      struct timeval tvE, tvB;
      gettimeofday( &tvB, NULL);
      nnHeapWrap( dr, dx, dh, dhi  );
      gettimeofday( &tvE, NULL);
      printf("time elapsed for nnHeapWrap : %6.2f \n", timeDiff(tvB,tvE) );

      numLeft -= pi;
      row += pi;
    }
    
    cudaMemcpy( &xmap.mat[IDX( i*RPI, 0, xmap.ld )], dhi.mat, sizeOfIntMatB(dhi), cudaMemcpyDeviceToHost );
  }
    
  cudaFree( dh.mat );
  cudaFree( dhi.mat );
  cudaFree( dx.mat );
  cudaFree( dr.mat );

  printf("done iterating.. copying over\n");
  cudaMemGetInfo(&memFree, &memTot);
  printf("GPU memory free = %lu/%lu (MB) \n",(unsigned long)memFree/(1024*1024),(unsigned long)memTot/(1024*1024));
  
  //DEBUG:
  copyAndMove( &rbcS->dx, &x );
  copyAndMoveI( &rbcS->dxMap, &xmap );
  rbcS->groupCount = (unint*)calloc( numReps, sizeof(unint) );
  for( i=0; i<numReps; i++ )
    rbcS->groupCount[i] = s;
  //END

  free( zeros.mat );
  free( xmap.mat );  
  free( max_h.mat );
}

#define RPI 1024 //reps per it
//s needs to be a multiple of 32, numReps a multiple of RPI
void buildBigOneShot( const hdMatrix x, vorStruct *vorS, unint numReps, unint s){
  unint i,j;
  unint n = x.r; 
  
  setupRepsVorHD( x, vorS, numReps );
  
  struct timeval tvB;
  gettimeofday(&tvB,NULL);
  sprintf(vorS->filename, "/local_data/lcayton/temp/%lu%lu.bin",tvB.tv_sec,tvB.tv_usec);
  vorS->map_fp = fopen(vorS->filename, "wb");
  
  intMatrix xMap;
  initIntMat( &xMap, RPI, s );
  xMap.mat = (unint*)calloc( sizeOfIntMat(xMap), sizeof(*xMap.mat) );
  
  /* vorS->xMap = (unint**)calloc( numReps, sizeof(unint*) ); */
  /* for( i=0; i<numReps; i++ ) */
  /*   vorS->xMap[i] = (unint*)calloc( s, sizeof(unint) ); */

  vorS->groupCount = (unint*)calloc( PAD(numReps), sizeof(*vorS->groupCount) );
  for( i=0; i<numReps; i++ )
    vorS->groupCount[i] = PAD(s);

  matrix dr;
  initMat( &dr, RPI, x.c );
  checkErr( cudaMalloc( (void**)&dr.mat, sizeOfMatB(dr) ) );

  matrix zeros;
  initMat( &zeros, BLOCK_SIZE, x.c );
  zeros.mat = (real*)calloc( sizeOfMat(zeros), sizeof(*zeros.mat) );

  //assume all of x is in CPU RAM
  //see how much fits onto the GPU at once.
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (size_t)(((double)memFree)*MEM_USABLE);
  memFree -= RPI*s*(sizeof(unint)+sizeof(real)); //"heap"
  unint memPerPt = PAD(x.c)*sizeof(real) + RPI*(sizeof(real)+sizeof(unint));
  unint ptsAtOnce = MIN( DPAD(memFree/memPerPt), n );
  
  printf(" mem free for x = %6.2f, ptsAtOnce = %u \n", ((double)memFree)/(1024.0*1024.0), ptsAtOnce);
  //allocate the temp dx which is on the device.
  matrix dx;
  initMat( &dx, ptsAtOnce, x.c );
  checkErr( cudaMalloc( (void**)&dx.mat, sizeOfMatB( dx ) ) );

  //allocate the dist matrix
  matrix dD;
  intMatrix dI; 
  initMat( &dD, RPI, ptsAtOnce+s );
  initIntMat( &dI, RPI, ptsAtOnce+s );
  checkErr( cudaMalloc( (void**)&dD.mat, sizeOfMatB( dD ) ) );
  checkErr( cudaMalloc( (void**)&dI.mat, sizeOfIntMatB( dI ) ) );

  //matrix in main memory for x
  matrix xmem;
  initMat( &xmem, ptsAtOnce, x.c );
  xmem.mat = (real*)calloc( sizeOfMat(xmem), sizeof(*xmem.mat) );

  for( i=0; i<numReps/RPI; i++ ){
    printf(" iterating through reps %d / %d \n", i, numReps/RPI );
    //copy the appropriate rows into dr
    cudaMemcpy( dr.mat, &vorS->r.mat[IDX( i*RPI, 0, vorS->r.ld )], sizeOfMatB( dr ), cudaMemcpyHostToDevice );
    //init D to MAX_REALs
    setConstantWrap( dD, MAX_REAL );

    unint numLeft = n;
    unint row = 0; //current row of x
    unint pi, pip;
    unint its=0;
    while( numLeft > 0 ){
      pi = MIN( ptsAtOnce, numLeft );
      pip = PAD( pi );
      
      readBlock(xmem, 0, x, row, pi);
      cudaMemcpy( dx.mat, xmem.mat, sizeof(*xmem.mat)*pi*xmem.pc, cudaMemcpyHostToDevice );
      dx.r = pi; dx.pr = pip;
      
      if( pip-pi )
	cudaMemcpy( &dx.mat[IDX( pi, 0, dx.ld )], zeros.mat, sizeof(*zeros.mat)*(pip-pi)*dx.pc, cudaMemcpyHostToDevice );

      //compute distances between dr and dx.  store them starting in column s.
      //store the indices of the xs in dI, starting from row.
      offDistWrap( dr, dx, dD, dI, s, row );

      sortDists( dD, dI );
    
      numLeft -= pi;
      row += pi;
      printf("\t it %d finished; %d points left\n", its++, numLeft); 
    }
    
    for( j=0; j<RPI; j++ )
      cudaMemcpy( &xMap.mat[IDX( j, 0, xMap.ld )], &dI.mat[IDX( j, 0, dI.ld )], s*sizeof(unint), cudaMemcpyDeviceToHost );
    writeBlock( vorS->map_fp, xMap );
    
      //      cudaMemcpy( vorS->xMap[i*RPI+j], &dI.mat[IDX( j, 0, dI.ld )], s*sizeof(unint), cudaMemcpyDeviceToHost );
  }
  fclose( vorS->map_fp );
  
  free( zeros.mat );
  cudaFree( dr.mat );
  free( xmem.mat );
  cudaFree( dx.mat );
  cudaFree( dD.mat );
  cudaFree( dI.mat );
}


void buildRBC(const matrix x, rbcStruct *rbcS, unint numReps, unint s){
  unint n = x.pr;
  intMatrix xmap;

  setupReps(x, rbcS, numReps);
  copyAndMove(&rbcS->dx, &x);
  
  xmap.r=numReps; xmap.pr=PAD(numReps); xmap.c=s; xmap.pc=xmap.ld=PAD(s);
  xmap.mat = (unint*)calloc( xmap.pr*xmap.pc, sizeof(*xmap.mat) );
  copyAndMoveI(&rbcS->dxMap, &xmap);
  rbcS->groupCount = (unint*)calloc( PAD(numReps), sizeof(*rbcS->groupCount) );
  
  //Figure out how much fits into memory
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (unint)(((float)memFree)*MEM_USABLE);
  /* mem needed per rep:
   *  n*sizeof(real) - dist mat
   *  n*sizeof(char) - dir
   *  n*sizeof(int)  - dSums
   *  sizeof(real)   - dranges
   *  sizeof(int)    - dCnts
   *  MEM_USED_IN_SCAN - memory used internally
   */
  unint ptsAtOnce = DPAD(memFree/((n+1)*sizeof(real) + n*sizeof(char) + (n+1)*sizeof(unint) + 2*MEM_USED_IN_SCAN(n)));
  if(!ptsAtOnce){
    fprintf(stderr,"error: %lu is not enough memory to build the RBC.. exiting\n", (unsigned long)memFree);
    exit(1);
  }

  //Now set everything up for the scans
  matrix dD;
  dD.pr=dD.r=ptsAtOnce; dD.c=rbcS->dx.r; dD.pc=rbcS->dx.pr; dD.ld=dD.pc;
  checkErr( cudaMalloc( (void**)&dD.mat, dD.pr*dD.pc*sizeof(*dD.mat) ) );
  
  real *dranges;
  checkErr( cudaMalloc( (void**)&dranges, ptsAtOnce*sizeof(real) ) );

  charMatrix ir;
  ir.r=dD.r; ir.pr=dD.pr; ir.c=dD.c; ir.pc=dD.pc; ir.ld=dD.ld;
  ir.mat = (char*)calloc( ir.pr*ir.pc, sizeof(*ir.mat) );
  charMatrix dir;
  copyAndMoveC(&dir, &ir);

  intMatrix dSums; //used to compute memory addresses.
  dSums.r=dir.r; dSums.pr=dir.pr; dSums.c=dir.c; dSums.pc=dir.pc; dSums.ld=dir.ld;
  checkErr( cudaMalloc( (void**)&dSums.mat, dSums.pc*dSums.pr*sizeof(*dSums.mat) ) );

  unint *dCnts;
  checkErr( cudaMalloc( (void**)&dCnts, ptsAtOnce*sizeof(*dCnts) ) );
  
  //Do the scans to build the dxMap
  unint numLeft = rbcS->dr.r; //points left to process
  unint row = 0; //base row for iteration of while loop
  unint pi, pip; //pi=pts per it, pip=pad(pi)
  while( numLeft > 0 ){
    pi = MIN(ptsAtOnce, numLeft);  //points to do this iteration.
    pip = PAD(pi);
    dD.r = pi; dD.pr = pip; dir.r=pi; dir.pr=pip; dSums.r=pi; dSums.pr=pip;

    distSubMat(rbcS->dr, rbcS->dx, dD, row, pip); //compute the distance matrix
    findRangeWrap(dD, dranges, s);  //find an appropriate range
    rangeSearchWrap(dD, dranges, dir); //set binary vector for points in range
    sumWrap(dir, dSums);  //This and the next call perform the parallel compaction.
    buildMapWrap(rbcS->dxMap, dir, dSums, row);
    getCountsWrap(dCnts,dir,dSums);  //How many points are assigned to each rep?  It is not 
                                     //*exactly* s, which is why we need to compute this.
    cudaMemcpy( &rbcS->groupCount[row], dCnts, pi*sizeof(*rbcS->groupCount), cudaMemcpyDeviceToHost );
    
    numLeft -= pi;
    row += pi;
  }
  
  cudaFree(dCnts);
  free(ir.mat);
  free(xmap.mat);
  cudaFree(dranges);
  cudaFree(dir.mat);
  cudaFree(dSums.mat);
  cudaFree(dD.mat);
}


// Choose representatives and move them to device
void setupReps(matrix x, rbcStruct *rbcS, unint numReps){
  unint i;
  unint *randInds;
  randInds = (unint*)calloc( PAD(numReps), sizeof(*randInds) );
  subRandPerm(numReps, x.r, randInds);
  
  matrix r;
  r.r=numReps; r.pr=PAD(numReps); r.c=x.c; r.pc=r.ld=PAD(r.c); 
  r.mat = (real*)calloc( r.pr*r.pc, sizeof(*r.mat) );

  for(i=0;i<numReps;i++)
    copyVector(&r.mat[IDX(i,0,r.ld)], &x.mat[IDX(randInds[i],0,x.ld)], x.c);
  
  copyAndMove(&rbcS->dr, &r);

  free(randInds);
  free(r.mat);
}


// Choose representatives and move them to device
void setupRepsVor(matrix x, vorStruct *vorS, unint numReps){
  unint i;
  unint *randInds;
  randInds = (unint*)calloc( PAD(numReps), sizeof(*randInds) );
  subRandPerm(numReps, x.r, randInds);
  
  initMat( &vorS->r, numReps, x.c );
  vorS->r.mat = (real*)calloc( sizeOfMat(vorS->r), sizeof(*(vorS->r.mat)) );

  for(i=0;i<numReps;i++)
    copyVector(&vorS->r.mat[IDX(i,0,vorS->r.ld)], &x.mat[IDX(randInds[i],0,x.ld)], x.c);
  
  free(randInds);

}



// Choose representatives and move them to device
//  x is on the HD.
void setupRepsVorHD(hdMatrix x, vorStruct *vorS, unint numReps){
  unint i;
  unint *randInds;
  randInds = (unint*)calloc( PAD(numReps), sizeof(*randInds) );
  subRandPerm(numReps, x.r, randInds);

  initMat( &vorS->r, numReps, x.c );
  vorS->r.mat = (real*)calloc( sizeOfMat(vorS->r), sizeof(*(vorS->r.mat)) );
  
  for(i=0;i<numReps;i++)
    readBlock( vorS->r, i, x, randInds[i], 1 );

  free(randInds);
}



//Assign each point in dq to its nearest point in dr.  
void computeReps(matrix dq, matrix dr, unint *repIDs, real *distToReps){
  real *dMins;
  unint *dMinIDs;

  checkErr( cudaMalloc((void**)&(dMins), dq.pr*sizeof(*dMins)) );
  checkErr( cudaMalloc((void**)&(dMinIDs), dq.pr*sizeof(*dMinIDs)) );
  
  nnWrap(dq,dr,dMins,dMinIDs);
  
  cudaMemcpy(distToReps,dMins,dq.r*sizeof(*dMins),cudaMemcpyDeviceToHost);
  cudaMemcpy(repIDs,dMinIDs,dq.r*sizeof(*dMinIDs),cudaMemcpyDeviceToHost);
  
  cudaFree(dMins);
  cudaFree(dMinIDs);
}


//Assumes radii is initialized to 0s
void computeRadii(unint *repIDs, real *distToReps, real *radii, unint n, unint numReps){
  unint i;

  for(i=0;i<n;i++)
    radii[repIDs[i]] = MAX(distToReps[i],radii[repIDs[i]]);
}


//Assumes groupCount is initialized to 0s
void computeCounts(unint *repIDs, unint n, unint *groupCount){
  unint i;
  
  for(i=0;i<n;i++)
    groupCount[repIDs[i]]++;
}


void buildQMap(matrix q, unint *qMap, unint *repIDs, unint numReps, unint *compLength){
  unint n=q.r;
  unint i;
  unint *gS; //groupSize
  
  gS = (unint*)calloc(numReps+1,sizeof(*gS));
  
  for( i=0; i<n; i++ )
    gS[repIDs[i]+1]++;
  for( i=0; i<numReps+1; i++ )
    gS[i]=PAD(gS[i]);
  
  for( i=1; i<numReps+1; i++ )
    gS[i]=gS[i-1]+gS[i];
  
  *compLength = gS[numReps];
  
  for( i=0; i<(*compLength); i++ )
    qMap[i]=DUMMY_IDX;
  
  for( i=0; i<n; i++ ){
    qMap[gS[repIDs[i]]]=i;
    gS[repIDs[i]]++;
  }

  free(gS);
}


// Sets the computation matrix to the identity.  
void idIntersection(charMatrix cM){
  unint i;
  for(i=0;i<cM.r;i++){
    if(i<cM.c)
      cM.mat[IDX(i,i,cM.ld)]=1;
  }
}


void fullIntersection(charMatrix cM){
  unint i,j;
  for(i=0;i<cM.r;i++){
    for(j=0;j<cM.c;j++){
      cM.mat[IDX(i,j,cM.ld)]=1;
    }
  }
}


void computeNNs(matrix dx, intMatrix dxMap, matrix dq, unint *dqMap, compPlan dcP, unint *NNs, real *NNdists, unint compLength){
  real *dNNdists;
  unint *dMinIDs;
  
  checkErr( cudaMalloc((void**)&dNNdists,compLength*sizeof(*dNNdists)) );
  checkErr( cudaMalloc((void**)&dMinIDs,compLength*sizeof(*dMinIDs)) );

  planNNWrap(dq, dqMap, dx, dxMap, dNNdists, dMinIDs, dcP, compLength );
  cudaMemcpy( NNs, dMinIDs, dq.r*sizeof(*NNs), cudaMemcpyDeviceToHost );
  cudaMemcpy( NNdists, dNNdists, dq.r*sizeof(*dNNdists), cudaMemcpyDeviceToHost );

  cudaFree(dNNdists);
  cudaFree(dMinIDs);
}


void computeKNNs(matrix dx, intMatrix dxMap, matrix dq, unint *dqMap, compPlan dcP, intMatrix NNs, matrix NNdists, unint compLength){
  matrix dNNdists;
  intMatrix dMinIDs;
  dNNdists.r=compLength; dNNdists.pr=compLength; dNNdists.c=KMAX; dNNdists.pc=KMAX; dNNdists.ld=dNNdists.pc;
  dMinIDs.r=compLength; dMinIDs.pr=compLength; dMinIDs.c=KMAX; dMinIDs.pc=KMAX; dMinIDs.ld=dMinIDs.pc;

  checkErr( cudaMalloc((void**)&dNNdists.mat,dNNdists.pr*dNNdists.pc*sizeof(*dNNdists.mat)) );
  checkErr( cudaMalloc((void**)&dMinIDs.mat,dMinIDs.pr*dMinIDs.pc*sizeof(*dMinIDs.mat)) );

  planKNNWrap(dq, dqMap, dx, dxMap, dNNdists, dMinIDs, dcP, compLength);
  cudaMemcpy( NNs.mat, dMinIDs.mat, dq.r*KMAX*sizeof(*NNs.mat), cudaMemcpyDeviceToHost );
  cudaMemcpy( NNdists.mat, dNNdists.mat, dq.r*KMAX*sizeof(*NNdists.mat), cudaMemcpyDeviceToHost );

  cudaFree(dNNdists.mat);
  cudaFree(dMinIDs.mat);
}


//This calls the dist1Kernel wrapper, but has it compute only 
//a submatrix of the all-pairs distance matrix.  In particular,
//only distances from dr[start,:].. dr[start+length-1] to all of x
//are computed, resulting in a distance matrix of size 
//length by dx.pr.  It is assumed that length is padded.
void distSubMat(matrix dr, matrix dx, matrix dD, unint start, unint length){
  dr.r=dr.pr=length;
  dr.mat = &dr.mat[IDX( start, 0, dr.ld )];
  dist1Wrap(dr, dx, dD);
}


void destroyRBC(rbcStruct *rbcS){
  cudaFree(rbcS->dx.mat);
  cudaFree(rbcS->dxMap.mat);
  cudaFree(rbcS->dr.mat);
  free(rbcS->groupCount);
}


void destroyVor(vorStruct *vorS){
    
  free( vorS->r.mat );
  free( vorS->groupCount );
  remove( vorS->filename );
}


/* Danger: this function allocates memory that it does not free.  
 * Use freeCompPlan to clear mem.  
 * See the readme.txt file for a description of why this function is needed.
 */
void initCompPlan(compPlan *dcP, charMatrix cM, unint *groupCountQ, unint *groupCountX, unint numReps){
  unint i,j,k;
  unint maxNumGroups=0;
  compPlan cP;
  
  unint sNumGroups = numReps;
  cP.numGroups = (unint*)calloc(sNumGroups, sizeof(*cP.numGroups));
  
  for(i=0; i<numReps; i++){
    cP.numGroups[i] = 0;
    for(j=0; j<numReps; j++)
      cP.numGroups[i] += cM.mat[IDX(i,j,cM.ld)];
    maxNumGroups = MAX(cP.numGroups[i], maxNumGroups);
  }
  cP.ld = maxNumGroups;
  
  unint sQToQGroup;
  for(i=0, sQToQGroup=0; i<numReps; i++)
    sQToQGroup += PAD(groupCountQ[i]);
  
  cP.qToQGroup = (unint*)calloc( sQToQGroup, sizeof(*cP.qToQGroup) );

  for(i=0, k=0; i<numReps; i++){
    for(j=0; j<PAD(groupCountQ[i]); j++)
      cP.qToQGroup[k++] = i;
  }
  
  unint sQGroupToXGroup = numReps*maxNumGroups;
  cP.qGroupToXGroup = (unint*)calloc( sQGroupToXGroup, sizeof(*cP.qGroupToXGroup) );
  unint sGroupCountX = maxNumGroups*numReps;
  cP.groupCountX = (unint*)calloc( sGroupCountX, sizeof(*cP.groupCountX) );
  
  for(i=0; i<numReps; i++){
    for(j=0, k=0; j<numReps; j++){
      if( cM.mat[IDX( i, j, cM.ld )] ){
	cP.qGroupToXGroup[IDX( i, k, cP.ld )] = j;
	cP.groupCountX[IDX( i, k++, cP.ld )] = groupCountX[j];
      }
    }
  }

  //Move to device
  checkErr( cudaMalloc( (void**)&dcP->numGroups, sNumGroups*sizeof(*dcP->numGroups) ) );
  cudaMemcpy( dcP->numGroups, cP.numGroups, sNumGroups*sizeof(*dcP->numGroups), cudaMemcpyHostToDevice );
  checkErr( cudaMalloc( (void**)&dcP->groupCountX, sGroupCountX*sizeof(*dcP->groupCountX) ) );
  cudaMemcpy( dcP->groupCountX, cP.groupCountX, sGroupCountX*sizeof(*dcP->groupCountX), cudaMemcpyHostToDevice );
  checkErr( cudaMalloc( (void**)&dcP->qToQGroup, sQToQGroup*sizeof(*dcP->qToQGroup) ) );
  cudaMemcpy( dcP->qToQGroup, cP.qToQGroup, sQToQGroup*sizeof(*dcP->qToQGroup), cudaMemcpyHostToDevice );
  checkErr( cudaMalloc( (void**)&dcP->qGroupToXGroup, sQGroupToXGroup*sizeof(*dcP->qGroupToXGroup) ) );
  cudaMemcpy( dcP->qGroupToXGroup, cP.qGroupToXGroup, sQGroupToXGroup*sizeof(*dcP->qGroupToXGroup), cudaMemcpyHostToDevice );
  dcP->ld = cP.ld;

  free(cP.numGroups);
  free(cP.groupCountX);
  free(cP.qToQGroup);
  free(cP.qGroupToXGroup);
}


//Frees memory allocated in initCompPlan.
void freeCompPlan(compPlan *dcP){
  cudaFree(dcP->numGroups);
  cudaFree(dcP->groupCountX);
  cudaFree(dcP->qToQGroup);
  cudaFree(dcP->qGroupToXGroup);
}


//sorts every row of dD with indices in dI 
void sortDists( matrix dD, intMatrix dI ){
  unint i;
  
  for( i=0; i<dD.r; i++ ){
    thrust::device_ptr<float> dPtr( &dD.mat[IDX( i, 0, dD.ld )] );
    thrust::device_ptr<unint> iPtr( &dI.mat[IDX( i, 0, dI.ld )] );
  
    thrust::sort_by_key( dPtr, dPtr+((size_t)dD.c), iPtr);
  }

}
#endif
