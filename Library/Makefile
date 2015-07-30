all: MoperatorCylwithZsym.o SourceGeneration.o MathTools.o MatVecMaker.o Output.o mympisetup.o filters.o ldos.o optfuncs.o shg.o

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules
CLEANFILES =  MoperatorCylwithZsym.o SourceGeneration.o MathTools.o MatVecMaker.o Output.o mympisetup.o filters.o ldos.o optfuncs.o shg.o

MoperatorCylwithZsym: MoperatorCylwithZsym.o
	$(CLINKER) -Wall -g  -o $@ MoperatorCylwithZsym.o 
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
optfuncs:optfuncs.o
	$(CLINKER) -Wall -g -o $@ optfuncs.o