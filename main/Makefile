all: optexec sfgexec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/opt/local/slepc-3.6.2
export NloptLibPath=/usr/local/lib/
export CavityLibPath=../Library

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLEANFILES = main.o sfg.o

optexec: main.o 
	$(CLINKER) main.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

opt: main.o
	$(CLINKER) -Wall -g -o $@ main.o 


sfgexec: sfg.o 
	$(CLINKER) sfg.o -o sfgexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

sfg: sfg.o
	$(CLINKER) -Wall -g -o $@ sfg.o 


