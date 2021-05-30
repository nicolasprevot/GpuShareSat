SMVER=50
EXEC      = test-glucose
DEPDIR    = rel-newtech/mtl rel-newtech/utils rel-newtech/core rel-newtech/simp rel-newtech/gpu rel-newtech/testUtils rel-newtech/satUtils gpuShareLib
LFLAGS = -l:libboost_unit_test_framework.a -l:libboost_test_exec_monitor.a -lpthread -arch=sm_52
COMP = nvcc
MROOT = $(PWD)/../..
include $(MROOT)/mtl/template.mk
