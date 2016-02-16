all: MoperatorCylwithZsym.o SourceGeneration.o MathTools.o MatVecMaker.o Output.o mympisetup.o filters.o ldos.o optfuncs.o shg.o shgcrosspol.o thg.o fieldfuncs.o eigsolver.o lvs1d.o sfg.o optfuncs2.o

export PETSC_DIR=/opt/local/petsc-3.6.3
export PETSC_ARCH=arch-opt
export SLEPC_DIR=/opt/local/slepc-3.6.2

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

CLEANFILES =  MoperatorCylwithZsym.o SourceGeneration.o MathTools.o MatVecMaker.o Output.o mympisetup.o filters.o ldos.o optfuncs.o shg.o shgcrosspol.o thg.o fieldfuncs.o eigsolver.o lvs1d.o sfg.o optfuncs2.o

MoperatorCylwithZsym: MoperatorCylwithZsym.o
	$(CLINKER) -Wall -g -o $@ MoperatorCylwithZsym.o 
SourceGeneration: SourceGeneration.o
	$(CLINKER) -Wall -g -o $@ SourceGeneration.o  
MathTools:MathTools.o
	$(CLINKER) -Wall -g -o $@ MathTools.o
MatVecMaker:MatVecMaker.o
	$(CLINKER) -Wall -g -o $@ MatVecMaker.o  
Output:Output.o
	$(CLINKER) -Wall -g -o $@ Output.o  
mympisetup:mympisetup.o
	$(CLINKER) -Wall -g -o $@ mympisetup.o
filters:filters.o
	$(CLINKER) -Wall -g -o $@ filters.o
ldos:ldos.o
	$(CLINKER) -Wall -g -o $@ ldos.o
shg:shg.o
	$(CLINKER) -Wall -g -o $@ shg.o
shgcrosspol:shgcrosspol.o
	$(CLINKER) -Wall -g -o $@ shgcrosspol.o
thg:thg.o
	$(CLINKER) -Wall -g -o $@ thg.o
fieldfuncs:fieldfuncs.o
	$(CLINKER) -Wall -g -o $@ fieldfuncs.o
optfuncs:optfuncs.o
	$(CLINKER) -Wall -g -o $@ optfuncs.o
eigsolver:eigsolver.o
	$(CLINKER) -Wall -g -o $@ eigsolver.o
lvs1d:lvs1d.o
	$(CLINKER) -Wall -g -o $@ lvs1d.o
sfg:sfg.o
	$(CLINKER) -Wall -g -o $@ sfg.o
optfuncs2:optfuncs2.o
	$(CLINKER) -Wall -g -o $@ optfuncs2.o
