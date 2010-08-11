#ifndef UTILSGPU_H
#define UTILSGPU_H

#include "defs.h"

memPlan createMemPlan(int,int);

void copyAndMove(matrix*,const matrix*);
void copyAndMoveI(intMatrix*,const intMatrix*);
void copyAndMoveC(charMatrix*,const charMatrix*);




#endif
