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
void readData(char*,unint,unint,real*);
void readDataText(char*,unint,unint,real*);
void orgData(real*,unint,unint,matrix,matrix);
void evalNNerror(matrix, matrix, unint*);
void evalKNNerror(matrix,matrix,intMatrix);

char *dataFile, *outFile;
unint n=0, m=0, d=0, numReps=0, s=0;
unint deviceNum=0;
int main(int argc, char**argv){
  real *data;
  matrix x, q;
  unint *NNs;
  intMatrix NNsK, kNNsRBC;
  unint i;
  struct timeval tvB,tvE;
  cudaError_t cE;
  rbcStruct rbcS;

  printf("*****************\n");
  printf("RANDOM BALL COVER\n");
  printf("*****************\n");
  
  parseInput(argc,argv);
  
  cuInit(0);
  printf("Using GPU #%d\n",deviceNum);
  if(cudaSetDevice(deviceNum) != cudaSuccess){
    printf("Unable to select device %d.. exiting. \n",deviceNum);
    exit(1);
  }
  
  unsigned int memFree, memTot;
  CUcontext pctx;
  unsigned int flags=0;
  int device;
  cudaGetDevice(&device);
  cuCtxCreate(&pctx,flags,device);
  cuMemGetInfo(&memFree, &memTot);
  printf("GPU memory free = %u/%u (MB) \n",memFree/(1024*1024),memTot/(1024*1024));

  data  = (real*)calloc( (n+m)*d, sizeof(*data) );
  x.mat = (real*)calloc( PAD(n)*PAD(d), sizeof(*(x.mat)) );

  //Need to allocate extra space, as each group of q will be padded later.
  q.mat = (real*)calloc( PAD(m)*PAD(d), sizeof(*(q.mat)) );
  x.r = n; x.c = d; x.pr = PAD(n); x.pc = PAD(d); x.ld = x.pc;
  q.r = m; q.c = d; q.pr = PAD(m); q.pc = PAD(d); q.ld = q.pc;

  NNs = (unint*)calloc( m, sizeof(*NNs) );
  for(i=0; i<m; i++)
    NNs[i]=DUMMY_IDX;
  
  readData(dataFile, (n+m), d, data);
  orgData(data, (n+m), d, x, q);
  free(data);

  NNsK.r=q.r; NNsK.pr=q.pr; NNsK.pc=NNsK.c=K; NNsK.ld=NNsK.pc;
  kNNsRBC.r=q.r; kNNsRBC.pr=q.pr; kNNsRBC.pc=kNNsRBC.c=K; kNNsRBC.ld=kNNsRBC.pc;
  kNNsRBC.mat = (unint*)calloc(kNNsRBC.pr*kNNsRBC.pc, sizeof(*kNNsRBC.mat));
  NNsK.mat = (unint*)calloc(NNsK.pr*NNsK.pc, sizeof(*NNsK.mat));
  
  /* printf("running k-brute force..\n"); */
  /* gettimeofday(&tvB,NULL); */
  /* bruteK(x,q,NNsK); */
  /* gettimeofday(&tvE,NULL); */
  /* printf("\t.. time elapsed = %6.4f \n",timeDiff(tvB,tvE)); */

  printf("\nrunning rbc..\n");
  gettimeofday(&tvB,NULL);
  buildRBC(x, &rbcS, numReps, s);
  gettimeofday(&tvE,NULL);
  printf("\t.. build time for rbc = %6.4f \n",timeDiff(tvB,tvE));

  //This finds the 32-NN; if you are only interested in the 1-NN, use queryRBC(..) instead
  gettimeofday(&tvB,NULL);
  kqueryRBC(q, rbcS, kNNsRBC);
  gettimeofday(&tvE,NULL);
  printf("\t.. query time for krbc = %6.4f \n",timeDiff(tvB,tvE));
  
  destroyRBC(&rbcS);
  printf("finished \n");
  
  cE = cudaGetLastError();
  if( cE != cudaSuccess ){
    printf("Execution failed; error type: %s \n", cudaGetErrorString(cE) );
  }
  
  evalKNNerror(x,q,kNNsRBC);

  free(NNs);
  free(NNsK.mat);
  free(kNNsRBC.mat);
  free(x.mat);
  free(q.mat);
}


void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -f datafile (bin) -n numPts (DB) -m numQueries -d dim -r numReps -s numPtsPerRep [-o outFile] [-g GPU num]\n\n");
    printf("\tdatafile     = binary file containing the data\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionailty\n");
    printf("\tnumReps      = number of representatives\n");
    printf("\tnumPtsPerRep = number of points assigned to each representative\n");
    printf("\toutFile      = output file (optional); stored in text format\n");
    printf("\tGPU num      = ID # of the GPU to use (optional) for multi-GPU machines\n");
    printf("\n\n");
    exit(0);
  }
  
  while(i<argc){
    if(!strcmp(argv[i], "-f"))
      dataFile = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-s"))
      s = atoi(argv[++i]);
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

  if( !n || !m || !d || !numReps || !s || !dataFile){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  
  if(numReps>n){
    fprintf(stderr,"can't have more representatives than points.. exiting\n");
    exit(1);
  }
}


void readData(char *dataFile, unint rows, unint cols, real *data){
  FILE *fp;
  unint numRead;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  numRead = fread(data,sizeof(real),rows*cols,fp);
  if(numRead != rows*cols){
    fprintf(stderr,"error reading file.. exiting \n");
    exit(1);
  }
  fclose(fp);
}


void readDataText(char *dataFile, unint rows, unint cols, real *data){
  FILE *fp;
  real t;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for(int i=0; i<rows; i++){
    for(int j=0; j<cols; j++){
      if(fscanf(fp,"%f ", &t)==EOF){
	fprintf(stderr,"error reading file.. exiting \n");
	exit(1);
      }
      data[IDX( i, j, cols )]=(real)t;
    }
  }
  fclose(fp);
}

//This function splits the data into two matrices, x and q, of 
//their specified dimensions.  The data is split randomly.
//It is assumed that the number of rows of data (the parameter n)
//is at least as large as x.r+q.r
void orgData(real *data, unint n, unint d, matrix x, matrix q){
   
  unint i,fi,j;
  unint *p;
  p = (unint*)calloc(n,sizeof(*p));
  
  randPerm(n,p);

  for(i=0,fi=0 ; i<x.r ; i++,fi++){
    for(j=0;j<x.c;j++){
      x.mat[IDX(i,j,x.ld)] = data[IDX(p[fi],j,d)];
    }
  }

  for(i=0 ; i<q.r ; i++,fi++){
    for(j=0;j<q.c;j++){
      q.mat[IDX(i,j,q.ld)] = data[IDX(p[fi],j,d)];
    } 
  }

  free(p);
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
    fprintf( fp, "%d %d %6.5f %6.5f \n", numReps, s, mean, sqrt(var) );
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
  
  gettimeofday(&tvB,NULL);
  bruteK(x,q,NNsB);
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
    fprintf( fp, "%d %d %6.5f %6.5f ", numReps, s, mean, sqrt(var) );
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
}
