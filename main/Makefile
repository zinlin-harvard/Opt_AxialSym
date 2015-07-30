all: optexec

export PETSC_DIR=/etc/petsc-3.5.3
export PETSC_ARCH=arch-opt
export CavityLibPath=../Library
export NloptLibPath=/usr/local/lib/

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES = main.o 

optexec: main.o 
	$(CLINKER) main.o -o optexec ${NloptLibPath}/libnlopt.a ${CavityLibPath}/libOPT.a $(PETSC_LIB)

opt: main.o
	$(CLINKER) -Wall -g -o $@ main.o 
