SMVER=52
EXEC      = perftest-glucose
DEPDIR    = glucose-syrup/mtl glucose-syrup/utils glucose-syrup/core glucose-syrup/simp glucose-syrup/gpu glucose-syrup/testUtils glucose-syrup/satUtils gpuShareLib
FLAG_COMPILE_O = -I/home/nicolas/programmation/libs/boost_1_61_0/
LFLAGS = -L/home/nicolas/programmation/libs/boost_1_61_0/stage/lib/ -l:libboost_unit_test_framework.a -l:libboost_test_exec_monitor.a -lpthread -arch=sm_$(SMVER)
COMP = nvcc
MROOT = $(PWD)/../..
include $(MROOT)/mtl/template.mk
