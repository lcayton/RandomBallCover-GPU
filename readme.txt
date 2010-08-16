***Random Ball Cover (RBC)***
Lawrence Cayton
lcayton@tuebingen.mpg.de

(C) Copyright 2010, Lawrence Cayton
 
This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------

This is a C and CUDA implementation of the Random Ball Cover data 
structure for nearest neighbor search.  

Notes on the code:
* The code requires that the entire DB and query set fit into the 
device memory.  

* For the most part, device variables (ie arrays residing on the GPU)
begin with a lowercase d.  For example, the device version of the 
DB variable x is dx.  

* The computePlan code is a bit more complex than is needed for the 
version of the RBC search algorithm described in the paper.  The 
search algorithm described in the paper has two steps: (1) Find the 
closest representative to the query.  (2) Explore the points owned
by that representative (ie the s-closest points to the representative
in the DB).  The computePlan code is more complex to make it easy
to try out other options.  For example, one could search the points
owned by the *two* closest representatives to the query instead.  This
would require only minor changes to the code.

* Currently the software works only in single precision.  If you wish
to switch to double precision, you must edit the defs.h file.  Simply 
uncomment the lines

typedef double real;
#define MAX_REAL DBL_MAX

and comment out the lines

typedef float real;
#define MAX_REAL FLT_MAX

Then, you must do a 

make clean

followed by another make.

* This software has been tested on the following graphics cards:
NVIDIA GTX 285
NVIDIA Tesla c2050.

* This sotware has been tested under the following software setup:
Ubuntu 9.10 (linux)
gcc 4.4
cuda 3.1

Please share your experience getting it to work under Windows and
Mac OSX!


