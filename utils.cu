#ifndef UTILS_CU
#define UTILS_CU

#include<sys/time.h>
#include<stdio.h>
#include "utils.h"
#include "defs.h"

//Returns a length l subset of a random permutation of [0,...,n-1]
//using the knuth shuffle.
//Input variable x is assumed to be alloced and of size l.
void subRandPerm(int l, int n, int *x){
  int i,ri, *y;
  y = (int*)calloc(n,sizeof(*y));
    
  struct timeval t3;
  gettimeofday(&t3,NULL);
  srand(t3.tv_usec);
  
  for(i=0;i<n;i++)
    y[i]=i;
  
  for(i=0;i<MIN(l,n-1);i++){  //The n-1 bit is necessary because you can't swap the last 
                              //element with something larger.
    ri=randBetween(i+1,n);
    swap(&y[i],&y[ri]);
  }
  
  for(i=0;i<l;i++)
    x[i]=y[i];
  free(y);
}

//Generates a random permutation of 0, ... , n-1 using the knuth shuffle.
//This should probably be merged with subRandPerm. 
void randPerm(int n, int *x){
  int i,ri;
  
  struct timeval t3;
  gettimeofday(&t3,NULL);
  srand(t3.tv_usec);
  
  for(i=0;i<n;i++){
    x[i]=i;
  }
  
  for(i=0;i<n-1;i++){
    ri=randBetween(i+1,n);
    swap(&x[i],&x[ri]);
  }
}

void swap(int *a, int *b){
  int t;
  t=*a; *a=*b; *b=t;
}

//generates a rand int in rand [a,b) 
int randBetween(int a, int b){
  int val,c;

  if(b<=a){
    fprintf(stderr,"misuse of randBetween.. exiting\n");
    exit(1);
  }
  c= b-a;

  while(c<= (val= rand()/(int)(((unsigned)RAND_MAX + 1) / c)));
  val=val+a;
  
  return val;
}


void printMat(matrix A){
  int i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%6.4f ",(float)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}


void printMatWithIDs(matrix A, int *id){
  int i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%6.4f ",(float)A.mat[IDX(i,j,A.ld)]);
    printf("%d ",id[i]);
    printf("\n");
  }
}


void printCharMat(charMatrix A){
  int i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%d ",(char)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}

void printIntMat(intMatrix A){
  int i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%d ",(int)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}

void printVector(real *x, int d){
  int i;

  for(i=0 ; i<d; i++)
    printf("%6.2f ",x[i]);
  printf("\n");
}


void copyVector(real *x, real *y, int d){
  int i;
  
  for(i=0;i<d;i++)
    x[i]=y[i];
}


void copyMat(matrix *x, matrix *y){
  int i,j;
  
  x->r=y->r; x->pr=y->pr; x->c=y->c; x->pc=y->pc; x->ld=y->ld;
  for(i=0; i<y->r; i++){
    for(j=0; j<y->c; j++){
      x->mat[IDX( i, j, x->ld )] = y->mat[IDX( i, j, y->ld )];
    }
  }
}


real distL1(matrix x, matrix y, int k, int l){
  int i;
  real ans=0;
  
  for(i=0;i<x.c;i++)
    ans+=abs(x.mat[IDX(k,i,x.ld)]-y.mat[IDX(l,i,x.ld)]);
  return ans;
}

double timeDiff(struct timeval start, struct timeval end){
  return (double)(end.tv_sec+end.tv_usec/1e6 - start.tv_sec - start.tv_usec/1e6); 
}



#endif
