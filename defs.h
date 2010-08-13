#ifndef DEFS_H
#define DEFS_H

#include<float.h>

#define FLOAT_TOL 1e-7
#define BLOCK_SIZE 16 //must be a power of 2

#define MAX_BS 65535 //max block size
#define SCAN_WIDTH 1024

// Format that the data is manipulated in:
typedef float real;
#define MAX_REAL FLT_MAX

// To switch to double precision, comment out the above 
// 2 lines and uncomment the following two lines. 

//typedef double real;
//#define MAX_REAL DBL_MAX

//Percentage of device mem to use
#define MEM_USABLE .95

#define DUMMY_IDX INT_MAX

//Row major indexing
#define IDX(i,j,ld) (((i)*(ld))+(j))

//increase an int to the next multiple of BLOCK_SIZE
#define PAD(i) ( ((i)%BLOCK_SIZE)==0 ? (i):((i)/BLOCK_SIZE)*BLOCK_SIZE+BLOCK_SIZE ) 

//decrease an int to the next multiple of BLOCK_SIZE
#define DPAD(i) ( ((i)%BLOCK_SIZE)==0 ? (i):((i)/BLOCK_SIZE)*BLOCK_SIZE ) 

#define MAX(i,j) ((i) > (j) ? (i) : (j))

#define MIN(i,j) ((i) < (j) ? (i) : (j))

typedef struct {
  real *mat;
  int r; //rows
  int c; //cols
  int pr; //padded rows
  int pc; //padded cols
  int ld; //the leading dimension (in this code, this is the same as pc)
} matrix;


typedef struct {
  char *mat;
  int r;
  int c;
  int pr;
  int pc;
  int ld;
} charMatrix;

typedef struct {
  int *mat;
  int r;
  int c;
  int pr;
  int pc;
  int ld;
} intMatrix;


typedef struct {
  int numComputeSegs;
  int normSegSize;//The number of points handled in one computation,
                     //though there will always be one leftover segment
                     //with (possibly) a different number of points.
  int lastSegSize;//.. and this is it.
} memPlan;


typedef struct{
  int *numGroups; //The number of groups of DB points to be examined.
  int sNumGroups; //size of the above array.
  int *groupCountX; //The number of elements in each group.
  int sGroupCountX;
  int *groupOff;
  int sGroupOff;
  int *qToGroup; //map from query to group #.
  int sQToGroup;
  int ld; //the width of memPos and groupCount (= max over numGroups)
} compPlan;
#endif
