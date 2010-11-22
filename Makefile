CC=gcc
NVCC=nvcc
CCFLAGS=
NVCCFLAGS= -DCUDA_FORCE_API_VERSION=3010
#other flags: -deviceemu -arch=sm_20 --ptxas-options=-v
#These are useful when debugging sometimes.  
LINKFLAGS=-lcuda  -lm
#other linkflags: -lgsl -lgslcblas 
# lgsl and lgslcblas are required if you want to use the GSL. 
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
