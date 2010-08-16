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
void readData(char*,int,int,real*);
void orgData(real*,int,int,matrix,matrix);


char *dataFile, *outFile;
int n=0, m=0, d=0, numReps=0, s=0;
int deviceNum=0;
int main(int argc, char**argv){
  real *data;
  matrix x, q;
  int *NNs, *NNsBrute;
  int i;
  struct timeval tvB,tvE;
  cudaError_t cE;

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

  NNs = (int*)calloc( m, sizeof(*NNs) );
  for(i=0; i<m; i++)
    NNs[i]=-1;
  NNsBrute = (int*)calloc( m, sizeof(*NNsBrute) );

  readData(dataFile, (n+m), d, data);
  orgData(data, (n+m), d, x, q);
  free(data);


  
  /* printf("db:\n"); */
  /* printMat(x); */
  /* printf("\nqueries: \n"); */
  /* printMat(q); */
  /* printf("\n\n"); */
  
  for(i=0;i<m;i++)
    NNs[i]=NNsBrute[i]=DUMMY_IDX;
  
  /* printf("running brute force..\n"); */
  /* gettimeofday(&tvB,NULL); */
  /* bruteSearch(x,q,NNsBrute); */
  /* gettimeofday(&tvE,NULL); */
  /* printf("\t.. time elapsed = %6.4f \n",timeDiff(tvB,tvE)); */
  
  
  printf("\nrunning rbc..\n");
  gettimeofday(&tvB,NULL);
  rbc(x,q,numReps,s,NNs); 
  gettimeofday(&tvE,NULL);
  printf("\t.. total time elapsed for rbc = %6.4f \n",timeDiff(tvB,tvE));
  printf("finished \n");
  
  cE = cudaGetLastError();
  if( cE != cudaSuccess ){
    printf("Execution failed; error type: %s \n", cudaGetErrorString(cE) );
  }

  printf("\nComputing error rates (this might take a while)\n");
  real *ranges = (real*)calloc(q.pr,sizeof(*ranges));
  for(i=0;i<q.r;i++)
    ranges[i] = distL1(q,x,i,NNs[i]) - 10e-6;
  

  int *cnts = (int*)calloc(q.pr,sizeof(*cnts));
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
  free(NNs);
  free(NNsBrute);
  free(x.mat);
  free(q.mat);
}


void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -f datafile (bin) -n numPts (DB) -m numQueries -d dim -r numReps -s numPtsPerRep [-o outFile] [-g GPU num]\n\n");
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


void readData(char *dataFile, int rows, int cols, real *data){
  FILE *fp;
  int numRead;

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


//This function splits the data into two matrices, x and q, of 
//their specified dimensions.  The data is split randomly.
//It is assumed that the number of rows of data (the parameter n)
//is at least as large as x.r+q.r
void orgData(real *data, int n, int d, matrix x, matrix q){
   
  int i,fi,j;
  int *p;
  p = (int*)calloc(n,sizeof(*p));
  
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

