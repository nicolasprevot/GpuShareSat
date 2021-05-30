SMVER=50
EXEC      = test-glucose
DEPDIR    = glucose-syrup/mtl glucose-syrup/utils glucose-syrup/core glucose-syrup/simp glucose-syrup/gpu glucose-syrup/testUtils glucose-syrup/satUtils gpuShareLib
LFLAGS = -l:libboost_unit_test_framework.a -l:libboost_test_exec_monitor.a -lpthread -arch=sm_52
COMP = nvcc
MROOT = $(PWD)/../..
include $(MROOT)/mtl/template.mk
