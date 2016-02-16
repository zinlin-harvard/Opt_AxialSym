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
#define __FUNCT__ "optsfg"
double optsfg(int DegFree, double *epsopt, double *grad, void *data)
{

  PetscErrorCode ierr;

  PetscPrintf(PETSC_COMM_WORLD,"********Entering the SFG FOM solver.********** \n");

  SFGdataGroup *ptdata = (SFGdataGroup *) data;

  double ldospowerindex = ptdata->ldospowerindex;
  double omega1a = ptdata->omega1a;
  double omega1b = ptdata->omega1b;
  double omega2 = ptdata->omega2;
  KSP ksp1a = ptdata->ksp1a;
  KSP ksp1b = ptdata->ksp1b;
  KSP ksp2 = ptdata->ksp2;
  int *its1a = ptdata->its1a;
  int *its1b = ptdata->its1b;
  int *its2 = ptdata->its2;
  Mat M1a = ptdata->M1a;
  Mat M1b = ptdata->M1b;
  Mat M2 = ptdata->M2;
  Vec b1a = ptdata->b1a;
  Vec b1b = ptdata->b1b;
  Vec x1a = ptdata->x1a;
  Vec x1b = ptdata->x1b;
  Vec ej = ptdata->ej;
  Vec J1aconj = ptdata->J1aconj;
  Vec J1bconj = ptdata->J1bconj;
  Vec epsSReal = ptdata->epsSReal;
  Vec epsFReal = ptdata->epsFReal;
  Vec epsDiff1a = ptdata->epsDiff1a;
  Vec epsDiff1b = ptdata->epsDiff1b;
  Vec epsDiff2 = ptdata->epsDiff2;
  Vec epsMed1a = ptdata->epsMed1a;
  Vec epsMed1b = ptdata->epsMed1b;
  Vec epsMed2 = ptdata->epsMed2;
  Vec epscoef1a = ptdata->epscoef1a;
  Vec epscoef1b = ptdata->epscoef1b;
  Vec epscoef2 = ptdata->epscoef2;
  Vec ldos1agrad = ptdata->ldos1agrad;
  Vec ldos1bgrad = ptdata->ldos1bgrad;
  Vec betagrad = ptdata->betagrad;
  int outputbase = ptdata->outputbase;
  //Mat B = ptdata->B;
  Vec vecNL = ptdata->vecNL;
  
  Vec x2;
  VecDuplicate(vR,&x2);
  
  Vec epsgrad,tmpgrad,fomgrad;
  VecDuplicate(epsSReal,&epsgrad);
  VecDuplicate(epsSReal,&tmpgrad);
  VecDuplicate(epsSReal,&fomgrad);
  RegzProj(DegFree,epsopt,epsSReal,epsgrad,pSIMP,bproj,etaproj,kspH,Hfilt,&itsH);

  MatMult(A,epsSReal,epsFReal);
  // Update the diagonals of M;                                                                                                                
  Mat MoneA, MoneB, Mtwo;
  MatDuplicate(M1a,MAT_COPY_VALUES,&MoneA);
  MatDuplicate(M1b,MAT_COPY_VALUES,&MoneB);
  MatDuplicate(M2,MAT_COPY_VALUES,&Mtwo);
  ModifyMatDiag(MoneA, D, epsFReal, epsDiff1a, epsMed1a, vecQ, omega1a, Nr, 1, Nz);
  ModifyMatDiag(MoneB, D, epsFReal, epsDiff1b, epsMed1b, vecQ, omega1b, Nr, 1, Nz);
  ModifyMatDiag(Mtwo, D, epsFReal, epsDiff2, epsMed2, vecQ, omega2, Nr, 1, Nz);

  double ldos1a=computeldos(ksp1a,MoneA,omega1a,epsFReal,b1a,J1aconj,x1a,epscoef1a,ldos1agrad,its1a);
  double ldos1b=computeldos(ksp1b,MoneB,omega1b,epsFReal,b1b,J1bconj,x1b,epscoef1b,ldos1bgrad,its1b);
  Vec ldosgrad;
  VecDuplicate(epsgrad,&ldosgrad);
  VecSet(ldosgrad,0);
  VecAXPY(ldosgrad,ldos1a,ldos1bgrad);
  VecAXPY(ldosgrad,ldos1b,ldos1agrad);
  double beta;
  beta = computesfg(x1a, x1b, x2, ej, its2, ksp1a, ksp1b, ksp2, MoneA, MoneB, Mtwo, omega2, epsFReal, epscoef1a, epscoef1b, epscoef2, betagrad, vecNL);
  
  double fom=beta/pow(ldos1a * ldos1b,ldospowerindex);

  VecScale(betagrad,1.0/pow(ldos1a * ldos1b,ldospowerindex));
  VecWAXPY(tmpgrad,-1.0*ldospowerindex*beta/pow(ldos1a * ldos1b,ldospowerindex+1.0),ldosgrad,betagrad);

  PetscPrintf(PETSC_COMM_WORLD,"----********the current fom at step %.5d is %.16e \n",count,fom);
  
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

  MatDestroy(&MoneA);
  MatDestroy(&MoneB);
  MatDestroy(&Mtwo);
  VecDestroy(&x2);
  VecDestroy(&epsgrad);
  VecDestroy(&ldosgrad);
  VecDestroy(&tmpgrad);
  VecDestroy(&fomgrad);
  count++;

  return fom;

}
