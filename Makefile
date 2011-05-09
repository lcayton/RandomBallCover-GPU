CC=gcc
NVCC=nvcc
NVCCFLAGS=-O3
# sometimes useful flags: -arch=sm_20 --ptxas-options=-v -g -G
# Note that you will need to specify an arch (as above) if you wish to use
# double precision
LINKFLAGS=-lcuda -lm
CUSOURCES=driver.cu utils.cu utilsGPU.cu rbc.cu brute.cu kernels.cu kernelWrap.cu sKernel.cu sKernelWrap.cu
CUOBJECTS=$(CUSOURCES:.cu=.o)
EXECUTABLE=testRBC
all: $(CUSOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(CUOBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(CUOBJECTS) -o $@ $(LINKFLAGS)

%.o:%.c
	$(NVCC) $(NVCCFLAGS) -c $+ 

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $+

clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
