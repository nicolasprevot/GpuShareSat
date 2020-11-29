##
##  Template makefile for Standard, Profile, Debug, Release, and Release-static versions
##
##    eg: "make rs" for a statically linked release version.
##        "make d"  for a debug version (no optimizations).
##        "make"    for the standard version (optimized, but with debug information and assertions active)
MAKEFLAGS += --no-builtin-rules
SMVER=52

SHELL=/bin/bash

PWD        = $(shell pwd)
EXEC      ?= $(notdir $(PWD))

CSRCS      = $(wildcard $(PWD)/*.cc) 
CUSRCS      = $(wildcard $(PWD)/*.cu) 
DSRCS      = $(foreach dir, $(DEPDIR), $(filter-out $(MROOT)/$(dir)/Main.cc, $(wildcard $(MROOT)/$(dir)/*.cc)))
DUSRCS      = $(foreach dir, $(DEPDIR), $(filter-out $(MROOT)/$(dir)/Main.cu, $(wildcard $(MROOT)/$(dir)/*.cu)))

CHDRS      = $(wildcard $(PWD)/../*/*.h) $(wildcard $(PWD)/../*/*.cuh)

# A bit of explanation: nvcc only accepts to compile .o files
# So use .n.o .p.o .r.o to differentiate 
# Also: to know if a file should be compiled with nvcc or just g++: we add .cu or .cc before
# So we have .cc.n.o .cu.n.o ...

COBJS      = $(CSRCS:.cc=.cc.n.o) $(DSRCS:.cc=.cc.n.o) $(CUSRCS:.cu=.cu.n.o) $(DUSRCS:.cu=.cu.n.o)


PCOBJS     = $(COBJS:.n.o=.p.o)
DCOBJS     = $(COBJS:.n.o=.d.o)
RCOBJS     = $(COBJS:.n.o=.r.o)
 
#There should be a variable COMP for the final compiler
CCX       ?= g++
NVCC      ?= nvcc

CFLAGS    ?= -Wall -Wno-parentheses
CUFLAGS   ?= -x cu -arch=sm_$(SMVER)

COPTIMIZE ?= -O3

CU_AND_C_FLAGS    += -I$(MROOT) -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS
LFLAGS    += -lz

.PHONY : s p d r rs clean 

s:	$(EXEC)
p:	$(EXEC)_profile
d:	$(EXEC)_debug
r:	$(EXEC)_release
rs:	$(EXEC)_static

libs:	lib$(LIB)_standard.a
libp:	lib$(LIB)_profile.a
libd:	lib$(LIB)_debug.a
libr:	lib$(LIB)_release.a

## Compile options
%.n.o:			CU_AND_C_FLAGS +=$(COPTIMIZE) -g -D DEBUG
%.p.o:			CU_AND_C_FLAGS +=$(COPTIMIZE) -pg -g -D NDEBUG -DPROFILE
%.d.o:			CU_AND_C_FLAGS +=-O0 -g -D DEBUG
%.r.o:			CU_AND_C_FLAGS +=$(COPTIMIZE) -g -D NDEBUG
%.cu.d.o:		CU_AND_C_FLAGS += -G -g
%.cu.n.o:		CU_AND_C_FLAGS += -G -g

## Link options
$(EXEC):		LFLAGS += -g
$(EXEC)_profile:	LFLAGS += -g -pg -l:libnvToolsExt.so -Xcompiler '-no-pie'
$(EXEC)_debug:		LFLAGS += -g
#$(EXEC)_release:	LFLAGS += ...
$(EXEC)_static:		LFLAGS += --static

## Dependencies
$(EXEC):		$(COBJS)
$(EXEC)_profile:	$(PCOBJS)
$(EXEC)_debug:		$(DCOBJS)
$(EXEC)_release:	$(RCOBJS)
$(EXEC)_static:		$(RCOBJS)

lib$(LIB)_standard.a:	$(filter-out */Main.*.n.o,  $(COBJS))
lib$(LIB)_profile.a:	$(filter-out */Main.*.p.o, $(PCOBJS))
lib$(LIB)_debug.a:	$(filter-out */Main.*.d.o, $(DCOBJS))
lib$(LIB)_release.a:	$(filter-out */Main.*.r.o, $(RCOBJS))

## Build rule
%.cc.n.o %.cc.p.o %.cc.d.o %.cc.r.o : %.cc
	@echo Compiling: $(subst $(MROOT)/,,$@)
	@$(CCX) $(CFLAGS) $(CU_AND_C_FLAGS) -c -o $@ $<

%.cu.n.o %.cu.p.o %.cu.d.o %.cu.r.o : %.cu
	@echo Compiling: $(subst $(MROOT)/,,$@)
	@$(NVCC) $(CUFLAGS) $(CU_AND_C_FLAGS) -dc -o $@ $<

## Linking rules (standard/profile/debug/release)
$(EXEC) $(EXEC)_profile $(EXEC)_debug $(EXEC)_release $(EXEC)_static:
	@echo Linking: "$@ ( $(foreach f,$^,$(subst $(MROOT)/,,$f)) )"
	@$(COMP) $^ $(LFLAGS) -o $@

## Library rules (standard/profile/debug/release)
lib$(LIB)_standard.a lib$(LIB)_profile.a lib$(LIB)_release.a lib$(LIB)_debug.a:
	@echo Making library: "$@ ( $(foreach f,$^,$(subst $(MROOT)/,,$f)) )"
	@$(AR) -rcsv $@ $^

## Library Soft Link rule:
libs libp libd libr:
	@echo "Making Soft Link: $^ -> lib$(LIB).a"
	@ln -sf $^ lib$(LIB).a

## Clean rule
allclean: clean
	
	@rm -f ../simp/*.o ../core/*.o
clean:
	rm -f $(EXEC) $(EXEC)_profile $(EXEC)_debug $(EXEC)_release $(EXEC)_static \
	  $(COBJS) $(PCOBJS) $(DCOBJS) $(RCOBJS) *.core depend.mk 

FULLDEPDIRS	=	$(patsubst %, $(MROOT)/%, $(DEPDIR))

## Make dependencies
depend.mk: $(CSRCS) $(CUSRCS) $(DSRCS) $(DUSRCS) $(CHDRS)
	@echo Making dependencies;
	@echo "making depend.mk";
	@rm -f depend.mk
	@for dir in $(FULLDEPDIRS) $(PWD); do \
		cfiles=$$(find $$dir -type f -name '*.cc'); \
		if [[ ! -z "$$cfiles" ]]; then \
			$(CCX) $(CFLAGS) -I$(MROOT) \
			$$cfiles -MM | sed "s|\(.*\)\.o:|$$dir/\1.cc.n.o $$dir/\1.cc.r.o $$dir/\1.cc.d.o $$dir/\1.cc.p.o:|" >> depend.mk; \
		fi;\
		cufiles=$$(find $$dir -type f -name '*.cu'); \
		if [[ ! -z "$$cufiles" ]]; then \
			$(NVCC) $(CUFLAGS) -I$(MROOT) \
			$$cufiles -E -Xcompiler "-isystem /usr/local/cuda-9.0/targets/x86_64-linux/include -MM" | sed "s|\(.*\)\.o:|$$dir/\1.cu.n.o $$dir/\1.cu.r.o $$dir/\1.cu.d.o $$dir/\1.cu.p.o:|" >> depend.mk; \
		fi;\
	done;

-include $(MROOT)/mtl/config.mk
-include depend.mk
