SMVER=50
EXEC      = test-glucose
DEPDIR    = mtl utils glucose-syrup/core glucose-syrup/simp glucose-syrup/gpu glucose-syrup/testUtils satUtils gpuShareLib
LFLAGS = -l:libboost_unit_test_framework.a -l:libboost_test_exec_monitor.a -lpthread -arch=sm_52
COMP = nvcc
MROOT = $(PWD)/../..
include $(MROOT)/mtl/template.mk
