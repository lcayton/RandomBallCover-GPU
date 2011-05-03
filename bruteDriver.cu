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
void readDataOff(char *dataFile, matrix x, unint num, unint off);
void writeNeighbs(char*,char*,intMatrix,matrix);

char *dataFileX, *dataFileQ, *dataFileXtxt, *dataFileQtxt, *outFile, *outFiletxt;
char runBrute=0, runEval=0;
unint n=0, m=0, d=0, deviceNum=0;

#define PTS_PER_IT 1000000

int main(int argc, char**argv){
  matrix x, q;
  struct timeval tvB,tvE;
  cudaError_t cE;
  
  printf("*****************\n");
  printf("RANDOM BALL COVER\n");
  printf("*****************\n");
  
  parseInput(argc,argv);
  
  gettimeofday( &tvB, NULL );
  printf("Using GPU #%d\n",deviceNum);
  if(cudaSetDevice(deviceNum) != cudaSuccess){
    printf("Unable to select device %d.. exiting. \n",deviceNum);
    exit(1);
  }
  
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  printf("GPU memory free = %lu/%lu (MB) \n",(unsigned long)memFree/(1024*1024),(unsigned long)memTot/(1024*1024));
  gettimeofday( &tvE, NULL );
  printf(" init time: %6.2f \n", timeDiff( tvB, tvE ) );
  

  //Setup matrices
  initMat( &x, PTS_PER_IT, d );
  initMat( &q, m, d );
  x.mat = (real*)calloc( sizeOfMat(x), sizeof(*(x.mat)) );
  q.mat = (real*)calloc( sizeOfMat(q), sizeof(*(q.mat)) );
    
  //Load data 
  readData( dataFileX, x );
  readData( dataFileQ, q );
  
  intMatrix nnsBrute;
  matrix distsBrute;
  initIntMat( &nnsBrute, m, KMAX );
  nnsBrute.mat = (unint*)calloc( sizeOfIntMat(nnsBrute), sizeof(*nnsBrute.mat) );
  initMat( &distsBrute, m, KMAX );
  distsBrute.mat = (real*)calloc( sizeOfMat(distsBrute), sizeof(*distsBrute.mat) );
  
  printf("running k-brute force..\n");
  gettimeofday( &tvB, NULL );
  bruteK( x, q, nnsBrute, distsBrute );
  gettimeofday( &tvE, NULL );
  printf( "\t.. time elapsed = %6.4f \n", timeDiff(tvB,tvE) );
  
  cE = cudaGetLastError();
  if( cE != cudaSuccess ){
    printf("Execution failed; error type: %s \n", cudaGetErrorString(cE) );
  }
  
  unint it=0;
  unint numDone = PTS_PER_IT;
  while (numDone < n){
    unint thisIt = MIN( PTS_PER_IT, n-numDone );
    readDataOff( dataFileX, x, thisIt, numDone );
    x.r = thisIt;
    x.pr = PAD(thisIt);
    
    printf("running k-brute force it %d ..\n", it++);
    gettimeofday( &tvB, NULL );
    warmBruteK( x, q, numDone, nnsBrute, distsBrute );
    gettimeofday( &tvE, NULL );
    printf( "\t.. time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

    numDone += thisIt;
  }


  //check the results
  /* matrix x2; */
  /* initMat( &x2, n, d ); */
  /* x2.mat = (real*)calloc( sizeOfMat(x2), sizeof(*(x2.mat)) ); */
  /* readData( dataFileX, x2 ); */
  /* intMatrix nns2; */
  /* matrix dists2; */
  /* initIntMat( &nns2, m, KMAX ); */
  /* nns2.mat = (unint*)calloc( sizeOfIntMat(nns2), sizeof(*nns2.mat) ); */
  /* initMat( &dists2, m, KMAX ); */
  /* dists2.mat = (real*)calloc( sizeOfMat(dists2), sizeof(*dists2.mat) ); */

  /* bruteK( x2, q, nns2, dists2 ); */
  /* unint i,j; */
  /* for( i=0; i<m; i++){ */
  /*   for(j=0; j<KMAX; j++){ */
  /*     if( nns2.mat[IDX( i, j, nns2.ld )] != nnsBrute.mat[IDX( i, j, nnsBrute.ld )] ) */
  /* 	printf("%d %d %f %f \n",nns2.mat[IDX( i, j, nns2.ld )],nnsBrute.mat[IDX( i, j, nnsBrute.ld )], dists2.mat[IDX( i, j, dists2.ld )], distsBrute.mat[IDX( i, j, distsBrute.ld )] ); */
  /*   } */
  /* } */
  /* free(nns2.mat); */
  /* free(dists2.mat); */

  if( outFile || outFiletxt )
    writeNeighbs( outFile, outFiletxt, nnsBrute, distsBrute );

  cudaThreadExit();
  free( nnsBrute.mat );
  free( distsBrute.mat );
  free( x.mat );
  free( q.mat );
}


void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -x datafileX -q datafileQ  -n numPts (DB) -m numQueries -d dim [-o outFile] [-g GPU num] [-b] [-e]\n\n");
    printf("\tdatafileX    = binary file containing the database\n");
    printf("\tdatafileQ    = binary file containing the queries\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionality\n");
    printf("\toutFile      = binary output file (optional)\n");
    printf("\tGPU num      = ID # of the GPU to use (optional) for multi-GPU machines\n");
    printf("\n\tuse -b to run brute force in addition the RBC\n");
    printf("\tuse -e to run the evaluation routine (implicitly runs brute force)\n");
    printf("\n\n\tTo input/output data in text format (instead of bin), use the \n\t-X and -Q and -O switches in place of -x and -q and -o (respectively).\n");
    printf("\n\n");
    exit(0);
  }
  
  while(i<argc){
    if(!strcmp(argv[i], "-x"))
      dataFileX = argv[++i];
    else if(!strcmp(argv[i], "-q"))
      dataFileQ = argv[++i];
    else if(!strcmp(argv[i], "-X"))
      dataFileX = argv[++i];
    else if(!strcmp(argv[i], "-Q"))
      dataFileQ = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-o"))
      outFile = argv[++i];
    else if(!strcmp(argv[i], "-O"))
      outFiletxt = argv[++i];
    else if(!strcmp(argv[i], "-g"))
      deviceNum = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-b"))
      runBrute=1;
    else if(!strcmp(argv[i], "-e"))
      runEval=1;
    else{
      fprintf(stderr,"%s : unrecognized option.. exiting\n",argv[i]);
      exit(1);
    }
    i++;
  }

  if( !n || !m || !d ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (!dataFileX && !dataFileXtxt) || (!dataFileQ && !dataFileQtxt) ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (dataFileX && dataFileXtxt) || (dataFileQ && dataFileQtxt) ){
    fprintf(stderr,"you can only give one database file and one query file.. exiting\n");
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



void readDataOff(char *dataFile, matrix x, unint num, unint off){
  unint i;
  FILE *fp;
  unint numRead;

  fp = fopen(dataFile,"rb");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
  
  fseek( fp, ((long)off)*x.c*sizeof(real), SEEK_SET);
  for( i=0; i<num; i++ ){ //can't load everything in one fread
                           //because matrix is padded.
    numRead = fread( &x.mat[IDX( i, 0, x.ld )], sizeof(real), x.c, fp );
    if(numRead != x.c){
      fprintf(stderr,"error reading file.. exiting \n");
      exit(1);
    }
  }
  fclose(fp);
}


void writeNeighbs(char *file, char *filetxt, intMatrix NNs, matrix dNNs){
  unint i,j;
  
  if( filetxt ) { //write text

    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }
    
    for( i=0; i<m; i++ ){
      for( j=0; j<KMAX; j++ )
	fprintf( fp, "%u ", NNs.mat[IDX( i, j, NNs.ld )] );
      fprintf(fp, "\n");
    }
    
    for( i=0; i<m; i++ ){
      for( j=0; j<KMAX; j++ )
	fprintf( fp, "%f ", dNNs.mat[IDX( i, j, dNNs.ld )]); 
      fprintf(fp, "\n");
    }
    fclose(fp);
    
  }

  if( file ){ //write binary

    FILE *fp = fopen(file,"wb");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }
    
    for( i=0; i<m; i++ )
      fwrite( &NNs.mat[IDX( i, 0, NNs.ld )], sizeof(*NNs.mat), KMAX, fp );
    for( i=0; i<m; i++ )
      fwrite( &dNNs.mat[IDX( i, 0, dNNs.ld )], sizeof(*dNNs.mat), KMAX, fp );
    
    fclose(fp);
  }
}
