all: eigencylindrical_exec

export PETSC_DIR=/etc/petsc-3.5.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/etc/slepc-3.5.4

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common
CLEANFILES = eigencylindrical.o

eigencylindrical_exec: eigencylindrical.o 
	$(CLINKER) eigencylindrical.o -o eigencylindrical_exec ../Library/libOPT.a $(PETSC_LIB) $(SLEPC_LIB)

eigencylindrical: eigencylindrical.o
	$(CLINKER) -Wall -g -o $@ eigencylindrical.o
