all: eigencylindrical_exec

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/opt/local/slepc-3.6.2

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLEANFILES = eigencylindrical.o

eigencylindrical_exec: eigencylindrical.o 
	$(CLINKER) eigencylindrical.o -o eigencylindrical_exec ../Library/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigencylindrical: eigencylindrical.o
	$(CLINKER) -Wall -g -o $@ eigencylindrical.o
