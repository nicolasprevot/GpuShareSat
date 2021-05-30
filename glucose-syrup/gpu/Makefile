# for GTX 940MX
# NOTE: 52 is too high for this GPU
SMVER=50

#for Tesla K40
#SMVER=35

EXEC = glucose-gpu
DEPDIR    = mtl utils glucose-syrup/core glucose-syrup/simp satUtils gpuShareLib
MROOT = $(PWD)/../..
COMP = nvcc
CFLAGS = -std=c++11
CUFLAGS = -std=c++11 -arch=sm_${SMVER}
LFLAGS = -lpthread -arch=sm_$(SMVER)
include $(MROOT)/mtl/template.mk
