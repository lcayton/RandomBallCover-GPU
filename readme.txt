***Random Ball Cover (RBC) v0.2.4***
Lawrence Cayton
lcayton@tuebingen.mpg.de

(C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 
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
SUMMARY

This is a C and CUDA implementation of the Random Ball Cover data 
structure for fast nearest neighbor search on a GPU.  The code
implements the one-shot algorithm.

See the following papers for a detailed description of the search
algorithm and the theory behind it.

* L. Cayton, A nearest neighbor data structure for graphics hardware.
ADMS, 2010.
* L. Cayton, Accelerating nearest neighbor search on manycore systems.
Submitted, 2011. 


---------------------------------------------------------------------
COMPILATION

Type make in a shell.  Requires GCC and NVCC (CUDA).  The code has
been developed under GCC 4.4 and CUDA 3.1.


---------------------------------------------------------------------
USE

A sample driver is provided for the RBC.  To try it out, type
$ testRBC
at the prompt and a list of options will be displayed.

The output file format is a list of the queries' NNs,
followed by a list of the distances to those NNs.  Note that by
default, all input and output is stored in single-precision (float)
format.  

Basic functionality is provided through this driver, but I recommend
integrating the RBC code directly into your code for the best
results.  For many applications, the RBC needs to be built only once,
and then can be queried many times. 

The method requires a single parameter, the number of
representatives.  This parameter allows you to trade-off between
search quality and search speed.  The best way to set this parameter
is to try a few different values out; a good starting point is
generally 5*sqrt(n), where n is the number of database points.  Use
the eval option (-e) to print out the error rate.  See the paper
(Cayton, 2011) for detailed information on this parameter. 


---------------------------------------------------------------------
FILES

* brute.{h,cu} -- implementation of brute force search (CPU and GPU
  versions) 
* defs.h -- definitions of constants and macros, including the
  distance metric.
* driver.cu -- example code for using the RBC data structure.
* kernels.{h,cu} -- implementation of all the (device) kernel functions,
  except those related to the scan (see sKernels below)
* kernelWrap.{h,cu} -- CPU wrapper code around the kernels.
* rbc.{h,cu} -- the core of the RBC data structure.  Includes the
  implementation of build and search algorithms.
* sKernel.{h,cu} -- implementation of the kernel functions related to
  the parallel scan algorithm (used within the build method).
* sKernelWrap.{h,cu} -- wrappers for the kernels in sKernel.
* utils.{h,cu} -- misc utilities used in the code.
* utilsGPU.{h,cu} -- misc utilities related to the GPU.


---------------------------------------------------------------------
MISC NOTES ON THE CODE

* The code currently computes distance using the L_1 (manhattan)
  metric.  If you wish to use a different notion of distance, you must
  modify defs.h.  It is quite simple to switch to any metric that 
  operates alongs the coordinates independently (eg, any L_p metric),
  but more complex metrics will require some aditional work.  The L_2
  metric (standard Euclidean distance) is already defined in defs.h.  

* The k-NN code is currently hard-coded for k=32.  It is hard-coded
  because it uses a manually implemented sorting network. This design
  allows all sorting to take place in on-chip (shared) memory, and is
  highly efficient.  Note that the NNs are returned in sorted order,
  so that if one wants only, say, 5 NNs, one can simply ignore the
  last 27 returned indices.  For k>32, contact the author.

* The code requires that the entire DB and query set fit into the 
  device memory.  

* Currently the software works in single precision.  If you wish to 
  switch to double precision, you must edit the defs.h file.  Simply 
  uncomment the lines

typedef double real;
#define MAX_REAL DBL_MAX

and comment out the lines

typedef float real;
#define MAX_REAL FLT_MAX

Then, you must do a 
$ make clean
followed by another make.

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
  would require only minor changes to the code, though is currently 
  untested.

* This software has been tested on the following graphics cards:
  NVIDIA GTX 285, GT 430, GTX 480, GeForce 320M, Tesla c2050

* This sotware has been developed under the following software setup:
  Ubuntu 10.04 (linux)
  gcc 4.4
  cuda 3.2

  It has also been tested under Mac OSX.  Please share your
  experience getting it to work under Windows!
 
* If you are running this code on a GPU which is also driving your
  display: A well-known issue with CUDA code in this situation is that 
  a process within the operating system will automatically kill 
  kernels that have been running for more than 5-10 seconds or so.
  You can get around this in Linux by switching out of X-Windows (often 
  CTRL-ALT-F1 does the trick) and running the code directly from the
  terminal.
