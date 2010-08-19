CC=gcc
NVCC=nvcc
CCFLAGS=
NVCCFLAGS= --ptxas-options=-v
#other flags: -deviceemu -arch=sm_20 --compiler-bindir=/usr/bin/gcc-4.3
LINKFLAGS=-lcuda 
#other linkflags: 
SOURCES=
CUSOURCES= driver.cu utils.cu utilsGPU.cu rbc.cu brute.cu kernels.cu kernelWrap.cu sKernel.cu sKernelWrap.cu
OBJECTS=$(SOURCES:.c=.o)
CUOBJECTS=$(CUSOURCES:.cu=.o)
EXECUTABLE=testRBC
all: $(SOURCES) $(CUSOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(CUOBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(CUOBJECTS) -o $@ $(LINKFLAGS)

%.o:%.c
	$(NVCC) $(NVCCFLAGS) -c $+ 

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $+

clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
