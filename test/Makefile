EXEC      = test-glucose
DEPDIR    = mtl utils core simp gpu testUtils satUtils
LFLAGS = -l:libboost_unit_test_framework.a -l:libboost_test_exec_monitor.a -lpthread -arch=sm_60
COMP = nvcc
MROOT = $(PWD)/..
include $(MROOT)/mtl/template.mk
