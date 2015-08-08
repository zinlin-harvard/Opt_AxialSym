#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <complex.h>
#include <slepc.h>
#include "libOPT.h"

int mma_verbose;
//set up the global PML parameters
int mPML=2;
double Refl=1e-25;
/*------------------------------------------------------*/

int maxit=15;
int Nr, Nz, Mr, Mz, mr0, mz0, Npmlr, Npmlz, Mzr, Nzr, Mzslab, DegFree;
double hr, hz, hzr;
double omega, epsilon;
double epsair, epssub;
double Qabs;
Vec vR, vecRad, vecQ, epsSReal, epsFReal, epsDiff, epsmedium;
Mat A,B,C,D;
Mat Mopr;

int zbl;

/*------------------------------------------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;


  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

  /*-------Set up the options parameters-------------*/
  PetscBool flg;
  PetscOptionsGetInt(PETSC_NULL,"-Mr",&Mr,&flg);  MyCheckAndOutputInt(flg,Mr,"Mr","Mr");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-mr0",&mr0,&flg);  MyCheckAndOutputInt(flg,mr0,"mr0","mr0");
  PetscOptionsGetInt(PETSC_NULL,"-mz0",&mz0,&flg);  MyCheckAndOutputInt(flg,mz0,"mz0","mz0");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlr",&Npmlr,&flg);  MyCheckAndOutputInt(flg,Npmlr,"Npmlr","Npmlr");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");
  PetscOptionsGetInt(PETSC_NULL,"-NNr",&Nr,&flg);  MyCheckAndOutputInt(flg,Nr,"Nr","Nr");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");

  int mnum;
  PetscOptionsGetInt(PETSC_NULL,"-mnum",&mnum,&flg);  MyCheckAndOutputInt(flg,mnum,"mnum","mnum");

  PetscOptionsGetInt(PETSC_NULL,"-zbl",&zbl,&flg);  
  if(!flg) zbl=0;
  PetscPrintf(PETSC_COMM_WORLD,"-------Lower z boundary is 0(pml,default), 1(even) or -1(odd): %d \n",zbl);

  Mzr=Mr*Mz, Nzr=Nr*Nz;
  DegFree = (Mzslab==0)*Mr*Mz + (Mzslab==1)*Mr + (Mzslab==2)*Mz;
  PetscOptionsGetReal(PETSC_NULL,"-hr",&hr,&flg);  MyCheckAndOutputDouble(flg,hr,"hr","hr");
  hz=hr;
  hzr = hr*hz;

  int multiplier;
  PetscOptionsGetInt(PETSC_NULL,"-multiplier",&multiplier,&flg);
  if(!flg) multiplier=1;
  PetscPrintf(PETSC_COMM_WORLD,"------multiplier is: %d \n ",multiplier);
  
  double freq1;
  PetscOptionsGetReal(PETSC_NULL,"-freq1",&freq1,&flg);
  if(!flg) freq1=1.0;
  if(flg) MyCheckAndOutputDouble(flg,freq1,"freq1","freq1");
  omega=2.0*PI*freq1;

  PetscOptionsGetReal(PETSC_NULL,"-eps1",&epsilon,&flg);
  if(!flg) epsilon=3.0;
  if(flg) MyCheckAndOutputDouble(flg,epsilon,"epsilon","epsilon");

  PetscOptionsGetReal(PETSC_NULL,"-epsair",&epsair,&flg);
  if(!flg) epsair=1.0;
  if(flg) MyCheckAndOutputDouble(flg,epsair,"epsair","epsair");
  PetscOptionsGetReal(PETSC_NULL,"-epssub",&epssub,&flg);
  if(!flg) epssub=epsair;
  if(flg) MyCheckAndOutputDouble(flg,epssub,"epssub","epssub"); 

  Qabs=1.0/0.0;

  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  /*------Set up the A, C, D matrices--------------*/
  if(multiplier==1){
    myinterp(PETSC_COMM_WORLD,&A,Nr,Nz,Mr,Mz,mr0,mz0,Mzslab);
  }else{
    myinterpmultiplier(PETSC_COMM_WORLD,&A,Nr,Nz,multiplier,DegFree,mr0,mz0,Mzslab);
  }

  CongMat(PETSC_COMM_WORLD, &C, 6*Nzr);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nzr);

  int mA, nA;
  MatGetSize(A,&mA,&nA);
  PetscPrintf(PETSC_COMM_WORLD,"------the dimensions of A is %d by %d \n", mA, nA);

  /*-----Set up vR, vecRad ------*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nzr, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nzr);

  ierr = VecDuplicate(vR,&vecRad); CHKERRQ(ierr);
  GetRadiusVec(vecRad,Nr,Nz,hr,mnum);
  ierr = PetscObjectSetName((PetscObject) vecRad, "Radius");CHKERRQ(ierr);
  
  /*----Set up the universal parts of Mopr-------*/
  MoperatorCyl(PETSC_COMM_WORLD,&Mopr,mnum,omega);
  ierr = PetscObjectSetName((PetscObject) Mopr, "Mopr"); CHKERRQ(ierr);

  /*----Set up the weight vector to be printed and destroyed----*/
  Vec weight;
  VecDuplicate(vR,&weight); CHKERRQ(ierr);
  GetWeightVec(weight,Nr,Nz,zbl);
  ierr = PetscObjectSetName((PetscObject) weight, "wt"); CHKERRQ(ierr);
  OutputVec(PETSC_COMM_WORLD,weight,"weight",".m");
  VecDestroy(&weight);

  /*----Set up the vecQ, epsDiff and epsmedium vectors--------*/
  ierr = VecDuplicate(vR,&vecQ);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsDiff);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsmedium);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsmedium,  "epsmedium");CHKERRQ(ierr);

  MatMult(D,vR,vecQ);
  VecScale(vecQ,1.0/Qabs);
  VecAXPY(vecQ,1.0,vR);
  VecSet(epsDiff,epsilon);
  GetMediumVecwithSub(epsmedium,Nr,Nz,Mr,Mz,epsair,epssub,Mzslab);  

  /*-----Set up epsSReal, epsFReal, vgradlocal ------*/
  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);

  /*---------Setup the epsopt and grad arrays----------------*/
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  int i;
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  //set up the solver and solve;
  PetscPrintf(PETSC_COMM_WORLD,"---*****Performing the eigensolve*****-----\n");
  ArrayToVec(epsopt,epsSReal);
  MatMult(A,epsSReal,epsFReal);
  ModifyMatDiag(Mopr, D, epsFReal, epsDiff, epsmedium,  vecQ, omega, Nr, 1, Nz);

  VecPointwiseMult(epsFReal,epsFReal,epsDiff);
  VecAXPY(epsFReal,1.0,epsmedium);
  OutputVec(PETSC_COMM_WORLD, epsFReal, "epsF",".m");
  OutputVec(PETSC_COMM_WORLD, epsmedium, "epsmed",".m");
  OutputVec(PETSC_COMM_WORLD, vecRad, "vecRad",".m");

  VecPointwiseMult(epsFReal,epsFReal,vecQ);
  eigsolver(Mopr,epsFReal,D);

/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

/* ----------------------Destroy Vecs and Mats----------------------------*/ 
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&Mopr); CHKERRQ(ierr);  

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&vecQ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecRad); CHKERRQ(ierr);

  ierr = VecDestroy(&epsDiff); CHKERRQ(ierr);
  ierr = VecDestroy(&epsmedium); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);

  free(epsopt);
  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = SlepcFinalize(); CHKERRQ(ierr);

  return 0;
}

  
