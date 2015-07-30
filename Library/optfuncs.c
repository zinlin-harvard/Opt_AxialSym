#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

extern int count;
extern int Nr, Nz;
extern double hr, hz;
extern Mat A,C,D;
extern Vec vR, vecRad, vecQ, weight, vgradlocal;
extern VecScatter scatter;
extern IS from, to;
extern char filenameComm[PETSC_MAX_PATH_LEN];

extern int pSIMP;
extern double bproj, etaproj;
extern Mat Hfilt;
extern KSP kspH;
extern int itsH;

#undef __FUNCT__ 
#define __FUNCT__ "optldos"
double optldos(int DegFree, double *epsopt, double *grad, void *data)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the LDOS solver.********** \n");

  LDOSdataGroup *ptdata = (LDOSdataGroup *) data;

  double omega = ptdata->omega;
  KSP ksp = ptdata->ksp;
  int *its = ptdata->its;
  Mat M = ptdata->M;
  Vec b = ptdata->b;
  Vec x = ptdata->x;
  Vec Jconj = ptdata->Jconj;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  Vec epsDiff = ptdata->epsDiff;
  Vec epsMed = ptdata->epsMed;
  Vec epscoef = ptdata->epscoef;
  Vec ldosgrad = ptdata->ldosgrad;
  int outputbase = ptdata->outputbase;

  Vec epsgrad;
  VecDuplicate(epsSReal,&epsgrad);
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);  
  // Update the diagonals of M;
  Mat Mopr;
  MatDuplicate(M,MAT_COPY_VALUES,&Mopr);
  ModifyMatDiag(Mopr, D, epsFReal, epsDiff, epsMed, vecQ, omega, Nr, 1, Nz);
  
  double ldos=computeldos(ksp,Mopr,omega,epsFReal,b,Jconj,x,epscoef,ldosgrad,its);
  ierr=VecPointwiseMult(ldosgrad,ldosgrad,epsgrad); CHKERRQ(ierr);
  KSPSolveTranspose(kspH,ldosgrad,epsgrad);
  ierr = VecToArray(epsgrad,grad,scatter,from,to,vgradlocal,DegFree);

  if(count%outputbase==0)
    {
      char buffer[1000];
      FILE *epsFile,*dofFile;
      int i;
      double *tmpeps;
      tmpeps = (double *) malloc(DegFree*sizeof(double));
      ierr = VecToArray(epsSReal,tmpeps,scatter,from,to,vgradlocal,DegFree);

      sprintf(buffer,"%s_%.5deps.txt",filenameComm,count);
      epsFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
        fprintf(epsFile,"%0.16e \n",tmpeps[i]);}
      fclose(epsFile);

      sprintf(buffer,"%s_%.5ddof.txt",filenameComm,count);
      dofFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
        fprintf(dofFile,"%0.16e \n",epsopt[i]);}
      fclose(dofFile);
    }

  MatDestroy(&Mopr);
  VecDestroy(&epsgrad);
  count++;

  return ldos;

}

#undef __FUNCT__
#define __FUNCT__ "optfomshg"
double optfomshg(int DegFree, double *epsopt, double *grad, void *data)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the SHG FOM solver.********** \n");

  SHGdataGroup *ptdata = (SHGdataGroup *) data;

  double ldospowerindex = ptdata->ldospowerindex;
  double omega1 = ptdata->omega1;
  double omega2 = ptdata->omega2;
  KSP ksp1 = ptdata->ksp1;
  KSP ksp2 = ptdata->ksp2;
  int *its1 = ptdata->its1;
  int *its2 = ptdata->its2;
  Mat M1 = ptdata->M1;
  Mat M2 = ptdata->M2;
  Vec b1 = ptdata->b1;
  Vec x1 = ptdata->x1;
  Vec ej = ptdata->ej;
  Vec J1conj = ptdata->J1conj;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  Vec epsDiff1 = ptdata->epsDiff1;
  Vec epsDiff2 = ptdata->epsDiff2;
  Vec epsMed1 = ptdata->epsMed1;
  Vec epsMed2 = ptdata->epsMed2;
  Vec epscoef1 = ptdata->epscoef1;
  Vec epscoef2 = ptdata->epscoef2;
  Vec ldos1grad = ptdata->ldos1grad;
  Vec betagrad = ptdata->betagrad;
  int outputbase = ptdata->outputbase;

  Vec epsgrad,tmpgrad,fomgrad;
  VecDuplicate(epsSReal,&epsgrad);
  VecDuplicate(epsSReal,&tmpgrad);
  VecDuplicate(epsSReal,&fomgrad);
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);
  // Update the diagonals of M;                                                                                                                
  Mat Mone, Mtwo;
  MatDuplicate(M1,MAT_COPY_VALUES,&Mone);
  MatDuplicate(M2,MAT_COPY_VALUES,&Mtwo);
  ModifyMatDiag(Mone, D, epsFReal, epsDiff1, epsMed1, vecQ, omega1, Nr, 1, Nz);
  ModifyMatDiag(Mtwo, D, epsFReal, epsDiff2, epsMed2, vecQ, omega2, Nr, 1, Nz);

  double ldos1=computeldos(ksp1,Mone,omega1,epsFReal,b1,J1conj,x1,epscoef1,ldos1grad,its1);
  double beta=computebeta2(x1,ej,its2,ksp1,ksp2,Mone,Mtwo,omega1,omega2,epsFReal,epscoef1,epscoef2,betagrad);
  double fom=beta/pow(ldos1,ldospowerindex);
  PetscPrintf(PETSC_COMM_WORLD,"----********the current fom at step %.5d is %.16e \n",count,fom);

  VecScale(betagrad,1.0/pow(ldos1,ldospowerindex));
  VecWAXPY(tmpgrad,-1.0*ldospowerindex*beta/pow(ldos1,ldospowerindex+1.0),ldos1grad,betagrad);

  ierr=VecPointwiseMult(tmpgrad,tmpgrad,epsgrad); CHKERRQ(ierr);
  KSPSolveTranspose(kspH,tmpgrad,fomgrad);
  VecToArray(fomgrad,grad,scatter,from,to,vgradlocal,DegFree);

  if(count%outputbase==0)
    {
      char buffer[1000];
      FILE *epsFile,*dofFile;
      int i;
      double *tmpeps;
      tmpeps = (double *) malloc(DegFree*sizeof(double));
      ierr = VecToArray(epsSReal,tmpeps,scatter,from,to,vgradlocal,DegFree);

      sprintf(buffer,"%s_%.5deps.txt",filenameComm,count);
      epsFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
	fprintf(epsFile,"%0.16e \n",tmpeps[i]);}
      fclose(epsFile);

      sprintf(buffer,"%s_%.5ddof.txt",filenameComm,count);
      dofFile = fopen(buffer,"w");
      for (i=0;i<DegFree;i++){
        fprintf(dofFile,"%0.16e \n",epsopt[i]);}
      fclose(dofFile);
    }

  MatDestroy(&Mone);
  MatDestroy(&Mtwo);
  VecDestroy(&epsgrad);
  VecDestroy(&tmpgrad);
  VecDestroy(&fomgrad);
  count++;

  return fom;

}
