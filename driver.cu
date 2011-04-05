/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<math.h>
#include "defs.h"
#include "utils.h"
#include "utilsGPU.h"
#include "rbc.h"
#include "brute.h"
#include "sKernel.h"

void parseInput(int,char**);
void readData(char*,matrix);
void readDataText(char*,matrix);
void evalNNerror(matrix, matrix, unint*);
void evalKNNerror(matrix,matrix,intMatrix);

char *dataFileX, *dataFileQ, *outFile;
char runBrute=0, runEval=0;
unint n=0, m=0, d=0, numReps=0;
unint deviceNum=0;
int main(int argc, char**argv){
  matrix x, q;
  intMatrix nnsRBC;
  matrix distsRBC;
  struct timeval tvB,tvE;
  cudaError_t cE;
  rbcStruct rbcS;

  printf("*****************\n");
  printf("RANDOM BALL COVER\n");
  printf("*****************\n");
  
  parseInput(argc,argv);
  
  printf("Using GPU #%d\n",deviceNum);
  if(cudaSetDevice(deviceNum) != cudaSuccess){
    printf("Unable to select device %d.. exiting. \n",deviceNum);
    exit(1);
  }
  
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  printf("GPU memory free = %lu/%lu (MB) \n",(unsigned long)memFree/(1024*1024),(unsigned long)memTot/(1024*1024));

  initMat( &x, n, d );
  initMat( &q, m, d );
  x.mat = (real*)calloc( sizeOfMat(x), sizeof(*(x.mat)) );
  q.mat = (real*)calloc( sizeOfMat(q), sizeof(*(q.mat)) );
    
  //Load data 
  readData( dataFileX, x );
  readData( dataFileQ, q );

  //Allocate space for NNs and dists
  initIntMat( &nnsRBC, m, K );
  initMat( &distsRBC, m, K );
  nnsRBC.mat = (unint*)calloc( sizeOfIntMat(nnsRBC), sizeof(*nnsRBC.mat) );
  distsRBC.mat = (real*)calloc( sizeOfMat(distsRBC), sizeof(*distsRBC.mat) );

  printf("\nrunning rbc..\n");
  //Build the RBC
  gettimeofday(&tvB,NULL);
  buildRBC(x, &rbcS, numReps, numReps);
  gettimeofday(&tvE,NULL);
  printf("\t.. build time for rbc = %6.4f \n",timeDiff(tvB,tvE));
  
  //This finds the 32-NNs; if you are only interested in the 1-NN, use queryRBC(..) instead
  gettimeofday(&tvB,NULL);
  kqueryRBC(q, rbcS, nnsRBC, distsRBC);
  gettimeofday(&tvE,NULL);
  printf("\t.. query time for krbc = %6.4f \n",timeDiff(tvB,tvE));
  
  if( runBrute ){
    intMatrix nnsBrute;
    matrix distsBrute;
    initIntMat( &nnsBrute, m, K );
    nnsBrute.mat = (unint*)calloc( sizeOfIntMat(nnsBrute), sizeof(*nnsBrute.mat) );
    initMat( &distsBrute, m, K );
    distsBrute.mat = (real*)calloc( sizeOfMat(distsBrute), sizeof(*distsBrute.mat) );
    
    printf("running k-brute force..\n");
    gettimeofday(&tvB,NULL);
    bruteK(x,q,nnsBrute,distsBrute);
    gettimeofday(&tvE,NULL);
    printf("\t.. time elapsed = %6.4f \n",timeDiff(tvB,tvE));
    
    free(nnsBrute.mat);
    free(distsBrute.mat);
  }

  cE = cudaGetLastError();
  if( cE != cudaSuccess ){
    printf("Execution failed; error type: %s \n", cudaGetErrorString(cE) );
  }
  
  if( runEval )
    evalKNNerror(x,q,nnsRBC);
  
  destroyRBC(&rbcS);
  cudaThreadExit();
  free(nnsRBC.mat);
  free(distsRBC.mat);
  free(x.mat);
  free(q.mat);
}


void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -x datafileX -q datafileQ  -n numPts (DB) -m numQueries -d dim -r numReps [-o outFile] [-g GPU num] [-b] [-e]\n\n");
    printf("\tdatafileX    = binary file containing the database\n");
    printf("\tdatafileQ    = binary file containing the queries\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionailty\n");
    printf("\tnumReps      = number of representatives\n");
    printf("\toutFile      = output file (optional); stored in text format\n");
    printf("\tGPU num      = ID # of the GPU to use (optional) for multi-GPU machines\n");
    printf("\n\tuse -b to run brute force in addition the RBC\n");
    printf("\tuse -e option to run evaluation routine\n");
    printf("\n\n");
    exit(0);
  }
  
  while(i<argc){
    if(!strcmp(argv[i], "-x"))
      dataFileX = argv[++i];
    else if(!strcmp(argv[i], "-q"))
      dataFileQ = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-o"))
      outFile = argv[++i];
    else if(!strcmp(argv[i], "-g"))
      deviceNum = atoi(argv[++i]);
    else{
      fprintf(stderr,"%s : unrecognized option.. exiting\n",argv[i]);
      exit(1);
    }
    i++;
  }

  if( !n || !m || !d || !numReps || !dataFileX || !dataFileQ ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  
  if(numReps>n){
    fprintf(stderr,"can't have more representatives than points.. exiting\n");
    exit(1);
  }
}


void readData(char *dataFile, matrix x){
  unint i;
  FILE *fp;
  unint numRead;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for( i=0; i<x.r; i++ ){ //can't load everything in one fread
                           //because matrix is padded.
    numRead = fread( &x.mat[IDX( i, 0, x.ld )], sizeof(real), x.c, fp );
    if(numRead != x.c){
      fprintf(stderr,"error reading file.. exiting \n");
      exit(1);
    }
  }
  fclose(fp);
}


void readDataText(char *dataFile, matrix x){
  FILE *fp;
  real t;
  int i,j;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for(i=0; i<x.r; i++){
    for(j=0; j<x.c; j++){
      if(fscanf(fp,"%f ", &t)==EOF){
	fprintf(stderr,"error reading file.. exiting \n");
	exit(1);
      }
      x.mat[IDX( i, j, x.ld )]=(real)t;
    }
  }
  fclose(fp);
}


//find the error rate of a set of NNs, then print it.
void evalNNerror(matrix x, matrix q, unint *NNs){
  struct timeval tvB, tvE;
  unint i;

  printf("\nComputing error rates (this might take a while)\n");
  real *ranges = (real*)calloc(q.pr,sizeof(*ranges));
  for(i=0;i<q.r;i++){
    if(NNs[i]>n) printf("error");
    ranges[i] = distVec(q,x,i,NNs[i]) - 10e-6;
  }

  unint *cnts = (unint*)calloc(q.pr,sizeof(*cnts));
  gettimeofday(&tvB,NULL);
  bruteRangeCount(x,q,ranges,cnts);
  gettimeofday(&tvE,NULL);
  
  long int nc=0;
  for(i=0;i<m;i++){
    nc += cnts[i];
  }
  double mean = ((double)nc)/((double)m);
  double var = 0.0;
  for(i=0;i<m;i++) {
    var += (((double)cnts[i])-mean)*(((double)cnts[i])-mean)/((double)m);
  }
  printf("\tavg rank = %6.4f; std dev = %6.4f \n\n", mean, sqrt(var));
  printf("(range count took %6.4f) \n", timeDiff(tvB, tvE));
  
  if(outFile){
    FILE* fp = fopen(outFile, "a");
    fprintf( fp, "%d %6.5f %6.5f \n", numReps, mean, sqrt(var) );
    fclose(fp);
  }

  free(ranges);
  free(cnts);
}


//evals the error rate of k-nns
void evalKNNerror(matrix x, matrix q, intMatrix NNs){
  struct timeval tvB, tvE;
  unint i,j,k;

  unint m = q.r;
  printf("\nComputing error rates (this might take a while)\n");
  
  unint *ol = (unint*)calloc( q.r, sizeof(*ol) );
  
  intMatrix NNsB;
  NNsB.r=q.r; NNsB.pr=q.pr; NNsB.c=NNsB.pc=32; NNsB.ld=NNsB.pc;
  NNsB.mat = (unint*)calloc( NNsB.pr*NNsB.pc, sizeof(*NNsB.mat) );
  matrix distsBrute;
  distsBrute.r=q.r; distsBrute.pr=q.pr; distsBrute.c=distsBrute.pc=K; distsBrute.ld=distsBrute.pc;
  distsBrute.mat = (real*)calloc( distsBrute.pr*distsBrute.pc, sizeof(*distsBrute.mat) );

  gettimeofday(&tvB,NULL);
  bruteK(x,q,NNsB,distsBrute);
  gettimeofday(&tvE,NULL);

   //calc overlap
  for(i=0; i<m; i++){
    for(j=0; j<K; j++){
      for(k=0; k<K; k++){
	ol[i] += ( NNs.mat[IDX(i, j, NNs.ld)] == NNsB.mat[IDX(i, k, NNsB.ld)] );
      }
    }
  }

  long int nc=0;
  for(i=0;i<m;i++){
    nc += ol[i];
  }

  double mean = ((double)nc)/((double)m);
  double var = 0.0;
  for(i=0;i<m;i++) {
    var += (((double)ol[i])-mean)*(((double)ol[i])-mean)/((double)m);
  }
  printf("\tavg overlap = %6.4f/%d; std dev = %6.4f \n", mean, K, sqrt(var));

  FILE* fp;
  if(outFile){
    fp = fopen(outFile, "a");
    fprintf( fp, "%d %6.5f %6.5f ", numReps, mean, sqrt(var) );
  }

  real *ranges = (real*)calloc(q.pr,sizeof(*ranges));
  for(i=0;i<q.r;i++){
    ranges[i] = distVec(q,x,i,NNs.mat[IDX(i, K-1, NNs.ld)]);
  }
  
  unint *cnts = (unint*)calloc(q.pr,sizeof(*cnts));
  bruteRangeCount(x,q,ranges,cnts);
  
  nc=0;
  for(i=0;i<m;i++){
    nc += cnts[i];
  }
  mean = ((double)nc)/((double)m);
  var = 0.0;
  for(i=0;i<m;i++) {
    var += (((double)cnts[i])-mean)*(((double)cnts[i])-mean)/((double)m);
  }
  printf("\tavg actual rank of 32nd NN returned by the RBC = %6.4f; std dev = %6.4f \n\n", mean, sqrt(var));
  printf("(brute k-nn took %6.4f) \n", timeDiff(tvB, tvE));

  if(outFile){
    fprintf( fp, "%6.5f %6.5f \n", mean, sqrt(var) );
    fclose(fp);
  }

  free(cnts);
  free(ol);
  free(NNsB.mat);
  free(distsBrute.mat);
}
