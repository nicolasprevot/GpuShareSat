SMVER=35
EXEC = glucose-gpu
DEPDIR    = mtl utils core simp satUtils
MROOT = $(PWD)/..
COMP = nvcc
#LFLAGS = -lpthread -arch=sm_60
LFLAGS = -lpthread -arch=sm_$(SMVER)
include $(MROOT)/mtl/template.mk
