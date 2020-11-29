SMVER=32
EXEC = glucose-gpu
DEPDIR    = mtl utils core simp satUtils
MROOT = $(PWD)/..
COMP = nvcc
CFLAGS = -std=c++11
CUFLAGS = -std=gnu++1y
LFLAGS = -lpthread -arch=sm_$(SMVER)
include $(MROOT)/mtl/template.mk
