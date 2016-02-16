#include <stdlib.h>
#include <petsc.h>
#include <string.h>
#include <nlopt.h>
#include <complex.h>
#include "libOPT.h"

int count=1;
int maxit=15;
int mma_verbose;
//set up the global PML parameters
int mPML=2;
double Refl=1e-25;
/*------------------------------------------------------*/

int Nr, Nz, Npmlr, Npmlz;
int zbl;
double hr, hz;
double Qabs;
Mat A,B,C,D;
Vec vR, vecRad, weight, unitr, unitp, unitz, vecQ, vgradlocal;
VecScatter scatter;
IS from, to;
char filenameComm[PETSC_MAX_PATH_LEN];

int pSIMP=1;
double bproj, etaproj;
Mat Hfilt;
KSP kspH;
int itsH=100;
/*------------------------------------------------------*/

#undef __FUNCT__ 
#define __FUNCT__ "main" 
int main(int argc, char **argv)
{
  /* -------Initialize ------*/
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  PetscPrintf(PETSC_COMM_WORLD,"--------Initializing------ \n");
  PetscErrorCode ierr;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0) 
    mma_verbose=1;

  /*-------Set up the options parameters-------------*/
  PetscBool flg;
  int Mr, Mz, mr0, mz0, Mzslab;
  PetscOptionsGetInt(PETSC_NULL,"-Mr",&Mr,&flg);  MyCheckAndOutputInt(flg,Mr,"Mr","Mr");
  PetscOptionsGetInt(PETSC_NULL,"-Mz",&Mz,&flg);  MyCheckAndOutputInt(flg,Mz,"Mz","Mz");
  PetscOptionsGetInt(PETSC_NULL,"-mr0",&mr0,&flg);  MyCheckAndOutputInt(flg,mr0,"mr0","mr0");
  PetscOptionsGetInt(PETSC_NULL,"-mz0",&mz0,&flg);  MyCheckAndOutputInt(flg,mz0,"mz0","mz0");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlr",&Npmlr,&flg);  MyCheckAndOutputInt(flg,Npmlr,"Npmlr","Npmlr");
  PetscOptionsGetInt(PETSC_NULL,"-Npmlz",&Npmlz,&flg);  MyCheckAndOutputInt(flg,Npmlz,"Npmlz","Npmlz");
  PetscOptionsGetInt(PETSC_NULL,"-NNr",&Nr,&flg);  MyCheckAndOutputInt(flg,Nr,"Nr","Nr");
  PetscOptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg);  MyCheckAndOutputInt(flg,Nz,"Nz","Nz");
  PetscOptionsGetInt(PETSC_NULL,"-Mzslab",&Mzslab,&flg);  MyCheckAndOutputInt(flg,Mzslab,"Mzslab","Mzslab");
  PetscOptionsGetInt(PETSC_NULL,"-zbl",&zbl,&flg);  MyCheckAndOutputInt(flg,zbl,"zbl","zbl");

  int Nzr, DegFree;
  double hzr;
  Nzr=Nr*Nz;
  DegFree = (Mzslab==0)*Mr*Mz + (Mzslab==1)*Mr + (Mzslab==2)*Mz;
  PetscOptionsGetReal(PETSC_NULL,"-hr",&hr,&flg);  MyCheckAndOutputDouble(flg,hr,"hr","hr");
  PetscOptionsGetReal(PETSC_NULL,"-hz",&hz,&flg);  MyCheckAndOutputDouble(flg,hz,"hz","hz");
  hzr = hr*hz;

  PetscOptionsGetReal(PETSC_NULL,"-Qabs",&Qabs,&flg);  
  if(Qabs>1e15) Qabs=1.0/0.0;
  MyCheckAndOutputDouble(flg,Qabs,"Qabs","Qabs");

  int multiplier;
  PetscOptionsGetInt(PETSC_NULL,"-multiplier",&multiplier,&flg);
  if(!flg) multiplier=1;
  PetscPrintf(PETSC_COMM_WORLD,"------multiplier is: %d \n ",multiplier); 

  int outputbase;
  PetscOptionsGetInt(PETSC_NULL,"-outputbase",&outputbase,&flg); MyCheckAndOutputInt(flg,outputbase,"outputbase","outputbase");

  double Jmag;
  PetscOptionsGetReal(PETSC_NULL,"-Jmag",&Jmag,&flg);  MyCheckAndOutputDouble(flg,Jmag,"Jmag","Jmag");

  int ptsrcr, ptsrcz, ptsrcdir;
  PetscOptionsGetInt(PETSC_NULL,"-ptsrcr",&ptsrcr,&flg);  MyCheckAndOutputInt(flg,ptsrcr,"ptsrcr","ptsrcr");
  PetscOptionsGetInt(PETSC_NULL,"-ptsrcz",&ptsrcz,&flg);  MyCheckAndOutputInt(flg,ptsrcz,"ptsrcz","ptsrcz");
  PetscOptionsGetInt(PETSC_NULL,"-ptsrcdir",&ptsrcdir,&flg);  MyCheckAndOutputInt(flg,ptsrcdir,"ptsrcdir","ptsrcdir");
  
  int ptsrc2r, ptsrc2z, ptsrc2dir;
  PetscOptionsGetInt(PETSC_NULL,"-ptsrc2r",&ptsrc2r,&flg);
  if(!flg) ptsrc2r=ptsrcr;
  PetscOptionsGetInt(PETSC_NULL,"-ptsrc2z",&ptsrc2z,&flg); 
  if(!flg) ptsrc2z=ptsrcz;
  PetscOptionsGetInt(PETSC_NULL,"-ptsrc2dir",&ptsrc2dir,&flg);
  if(!flg) ptsrc2dir=ptsrcdir;
  PetscPrintf(PETSC_COMM_WORLD,"----ptsrc2r, ptsrc2z, and ptsrc2dir: %d, %d, %d \n", ptsrc2r, ptsrc2z, ptsrc2dir);

  
  char initialdatafile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-filenameprefix",filenameComm,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,filenameComm,"filenameComm","Filename prefix");
  PetscOptionsGetString(PETSC_NULL,"-initdatfile",initialdatafile,PETSC_MAX_PATH_LEN,&flg); MyCheckAndOutputChar(flg,initialdatafile,"initialdatafile","Inputdata file");

  int solver;
  PetscOptionsGetInt(PETSC_NULL,"-solver",&solver,&flg);  MyCheckAndOutputInt(flg,solver,"solver","LU Direct solver choice (0 PASTIX, 1 MUMPS, 2 SUPERLU_DIST)");

  PetscOptionsGetReal(PETSC_NULL,"-bproj",&bproj,&flg);
  if(flg) MyCheckAndOutputDouble(flg,bproj,"bproj","bproj");
  if(!flg) bproj=0;
  PetscOptionsGetReal(PETSC_NULL,"-etaproj",&etaproj,&flg);
  if(flg) MyCheckAndOutputDouble(flg,etaproj,"etaproj","etaproj");
  if(!flg) etaproj=0.5;

  double sH, nR;
  int dimH;
  PetscOptionsGetReal(PETSC_NULL,"-sH",&sH,&flg); MyCheckAndOutputDouble(flg,sH,"sH","sH");
  PetscOptionsGetReal(PETSC_NULL,"-nR",&nR,&flg); MyCheckAndOutputDouble(flg,nR,"nR","nR");
  PetscOptionsGetInt(PETSC_NULL,"-dimH",&dimH,&flg); MyCheckAndOutputDouble(flg,dimH,"dimH","dimH");

  int m1a, m1b,m2;
  PetscOptionsGetInt(PETSC_NULL,"-m1a",&m1a,&flg);  MyCheckAndOutputInt(flg,m1a,"m1a","m1a");
  PetscOptionsGetInt(PETSC_NULL,"-m1b",&m1b,&flg);  MyCheckAndOutputInt(flg,m1b,"m1b","m1b");
  PetscOptionsGetInt(PETSC_NULL,"-m2",&m2,&flg);  MyCheckAndOutputInt(flg,m2,"m2","m2");

  double freq1a, freq1b, freq2;
  PetscOptionsGetReal(PETSC_NULL,"-freq1a",&freq1a,&flg); MyCheckAndOutputDouble(flg,freq1a,"freq1a","freq1a");
  PetscOptionsGetReal(PETSC_NULL,"-freq1b",&freq1b,&flg); MyCheckAndOutputDouble(flg,freq1b,"freq1b","freq1b");
  PetscOptionsGetReal(PETSC_NULL,"-freq2",&freq2,&flg); MyCheckAndOutputDouble(flg,freq2,"freq2","freq2");
  double omega1a, omega1b, omega2;
  omega1a=2.0*PI*freq1a;
  omega1b=2.0*PI*freq1b;
  omega2=2.0*PI*freq2;

  double epsilon1a, epsilon1b, epsilon2, epsair, epssub1a, epssub1b, epssub2;
  PetscOptionsGetReal(PETSC_NULL,"-eps1a",&epsilon1a,&flg); MyCheckAndOutputDouble(flg,epsilon1a,"epsilon1a","epsilon1a");
  PetscOptionsGetReal(PETSC_NULL,"-eps1b",&epsilon1b,&flg); MyCheckAndOutputDouble(flg,epsilon1b,"epsilon1b","epsilon1b");
  PetscOptionsGetReal(PETSC_NULL,"-eps2",&epsilon2,&flg); MyCheckAndOutputDouble(flg,epsilon2,"epsilon2","epsilon2"); 
  PetscOptionsGetReal(PETSC_NULL,"-epsair",&epsair,&flg); MyCheckAndOutputDouble(flg,epsair,"epsair","epsair");
  PetscOptionsGetReal(PETSC_NULL,"-epssub1a",&epssub1a,&flg); MyCheckAndOutputDouble(flg,epssub1a,"epssub1a","epssub1a"); 
  PetscOptionsGetReal(PETSC_NULL,"-epssub1b",&epssub1b,&flg); MyCheckAndOutputDouble(flg,epssub1b,"epssub1b","epssub1b"); 
  PetscOptionsGetReal(PETSC_NULL,"-epssub2",&epssub2,&flg); MyCheckAndOutputDouble(flg,epssub2,"epssub2","epssub2"); 

  char epsmed1afile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-epsmed1afilename",epsmed1afile,PETSC_MAX_PATH_LEN,&flg);
  if(!flg) strcpy(epsmed1afile,"");
  if(strcmp(epsmed1afile,"")==0)
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed1afile is  empty. \n");
  else
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed1afile is %s \n",epsmed1afile);

  char epsmed1bfile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-epsmed1bfilename",epsmed1bfile,PETSC_MAX_PATH_LEN,&flg);
  if(!flg) strcpy(epsmed1bfile,"");
  if(strcmp(epsmed1bfile,"")==0)
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed1bfile is  empty. \n");
  else
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed1bfile is %s \n",epsmed1bfile);

  char epsmed2file[PETSC_MAX_PATH_LEN];  
  PetscOptionsGetString(PETSC_NULL,"-epsmed2filename",epsmed2file,PETSC_MAX_PATH_LEN,&flg);
  if(!flg) strcpy(epsmed2file,"");
  if(strcmp(epsmed2file,"")==0)
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed2file is  empty. \n");
  else
    PetscPrintf(PETSC_COMM_WORLD,"----epsmed2file is %s \n",epsmed2file);
  
  char vecNLfile[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-vecNLfilename",vecNLfile,PETSC_MAX_PATH_LEN,&flg);
  if(!flg) strcpy(vecNLfile,"");
  if(strcmp(vecNLfile,"")==0)
    PetscPrintf(PETSC_COMM_WORLD,"----vecNLfile is empty. \n");
  else
    PetscPrintf(PETSC_COMM_WORLD,"----vecNLfile is %s \n",vecNLfile);
  
  /*------Set up the A, C, D matrices--------------*/
  if(multiplier==1){
    myinterp(PETSC_COMM_WORLD,&A,Nr,Nz,Mr,Mz,mr0,mz0,Mzslab);
  }else{
    myinterpmultiplier(PETSC_COMM_WORLD,&A,Nr,Nz,multiplier,Mr,Mz,mr0,mz0,Mzslab);
  }
  CongMat(PETSC_COMM_WORLD, &C, 6*Nzr);
  ImagIMat(PETSC_COMM_WORLD, &D,6*Nzr);

  /*-----Set up vR, vecRad, weight, unitr, unitp, unitz, vecQ-----*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 6*Nzr, &vR);CHKERRQ(ierr);
  GetRealPartVec(vR,6*Nzr);

  ierr = VecDuplicate(vR,&vecRad); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&weight); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitr); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitp); CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&unitz); CHKERRQ(ierr);
  GetWeightVec(weight,Nr,Nz,zbl);
  GetRadiusVec(vecRad,Nr,Nz,hr,m1a);
  GetUnitVec(unitr,0,6*Nzr);
  GetUnitVec(unitp,1,6*Nzr);
  GetUnitVec(unitz,2,6*Nzr);

  ierr = VecDuplicate(vR,&vecQ);CHKERRQ(ierr);
  MatMult(D,vR,vecQ);
  VecScale(vecQ,1.0/Qabs);
  VecAXPY(vecQ,1.0,vR);
  
  /*----Set up the universal parts of M1a, M1b and M2-------*/
  Mat M1a, M1b, M2;
  MoperatorCyl(PETSC_COMM_WORLD,&M1a,m1a,omega1a);
  MoperatorCyl(PETSC_COMM_WORLD,&M1b,m1b,omega1b);
  MoperatorCyl(PETSC_COMM_WORLD,&M2,m2,omega2);
  ierr = PetscObjectSetName((PetscObject) M1a, "M1a"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M1b, "M1b"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) M2, "M2"); CHKERRQ(ierr);

  /*----Set up the vecQ, epsDiff's, epscoef's and epsmedium vectors--------*/
  Vec epsDiff1a, epsDiff1b, epsDiff2, epsMed1a, epsMed1b, epsMed2;
  Vec epscoef1a, epscoef1b, epscoef2;
  ierr = VecDuplicate(vR,&epsDiff1a);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsDiff1b);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsDiff2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1a);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef1b);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epscoef2);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsMed1a);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsMed1b);CHKERRQ(ierr);
  ierr = VecDuplicate(vR,&epsMed2);CHKERRQ(ierr);

  VecSet(epsDiff1a,epsilon1a);
  VecSet(epsDiff1b,epsilon1b);
  VecSet(epsDiff2,epsilon2);

  VecPointwiseMult(epscoef1a,epsDiff1a,vecQ);
  VecPointwiseMult(epscoef1b,epsDiff1b,vecQ);
  VecScale(epscoef1a,omega1a*omega1a);
  VecScale(epscoef1b,omega1b*omega1b);
  VecPointwiseMult(epscoef2,epsDiff2,vecQ);
  VecScale(epscoef2,omega2*omega2);

  int i;
  if(strcmp(epsmed1afile,"")==0 && strcmp(epsmed1bfile,"")==0  && strcmp(epsmed2file,"")==0){
    GetMediumVecwithSub(epsMed1a,Nr,Nz,Mr,Mz,epsair,epssub1a,Mzslab,mr0,mz0);  
    GetMediumVecwithSub(epsMed1b,Nr,Nz,Mr,Mz,epsair,epssub1b,Mzslab,mr0,mz0);  
    GetMediumVecwithSub(epsMed2,Nr,Nz,Mr,Mz,epsair,epssub2,Mzslab,mr0,mz0);  
  }else{
    double *epsmed1Aarray, *epsmed1Barray, *epsmed2array;
    FILE *med1afile, *med1bfile, *med2file;
    epsmed1Aarray = (double *)malloc(6*Nz*Nr*sizeof(double));
    epsmed1Barray = (double *)malloc(6*Nz*Nr*sizeof(double));
    epsmed2array = (double *)malloc(6*Nz*Nr*sizeof(double));
    med1afile=fopen(epsmed1afile,"r");
    med1bfile=fopen(epsmed1bfile,"r");
    med2file=fopen(epsmed2file,"r");
    for(i=0;i<6*Nz*Nr;i++){
      fscanf(med1afile,"%lf",&epsmed1Aarray[i]);
      fscanf(med1bfile,"%lf",&epsmed1Barray[i]);
      fscanf(med2file,"%lf",&epsmed2array[i]);
    }
    ArrayToVec(epsmed1Aarray,epsMed1a);
    ArrayToVec(epsmed1Barray,epsMed1b);
    ArrayToVec(epsmed2array,epsMed2);
    fclose(med1afile);
    fclose(med1bfile);
    fclose(med2file);
    free(epsmed1Aarray);
    free(epsmed1Barray);
    free(epsmed2array);
  }

  Vec vecNL;
  VecDuplicate(vR,&vecNL);
  if(strcmp(vecNLfile,"")){
    FILE *filevecNL;
    double *vecNLarray;
    vecNLarray = (double *)malloc(6*Nz*Nr*sizeof(double));
    filevecNL=fopen(vecNLfile,"r");
    for(i=0;i<6*Nz*Nr;i++){
      fscanf(filevecNL,"%lf",&vecNLarray[i]);
    }
    ArrayToVec(vecNLarray,vecNL);
    fclose(filevecNL);
    free(vecNLarray);
  }
    
  
  /*-----Set up epsSReal, epsFReal, vgradlocal ------*/
  Vec epsSReal, epsFReal;
  ierr = MatCreateVecs(A,&epsSReal, &epsFReal); CHKERRQ(ierr);
  
  ierr = PetscObjectSetName((PetscObject) epsSReal, "epsSReal");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) epsFReal, "epsFReal");CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, DegFree, &vgradlocal); CHKERRQ(ierr);

  /*---------Set up ptsrc-------------*/
  Vec ptsrc;
  ierr = VecDuplicate(vR,&ptsrc);CHKERRQ(ierr);
  VecSet(ptsrc,0.0);
  if (ptsrcdir == 1)
    SourceSingleSetX(PETSC_COMM_WORLD, ptsrc, Nr, 1, Nz, ptsrcr, 0, ptsrcz,1.0/hzr);
  else if (ptsrcdir ==2)
    SourceSingleSetY(PETSC_COMM_WORLD, ptsrc, Nr, 1, Nz, ptsrcr, 0, ptsrcz,1.0/hzr);
  else if (ptsrcdir == 3)
    SourceSingleSetZ(PETSC_COMM_WORLD, ptsrc, Nr, 1, Nz, ptsrcr, 0, ptsrcz,1.0/hzr);
  else
    PetscPrintf(PETSC_COMM_WORLD," Please specify correct direction of point source current: x (1) , y (2) or z (3)\n "); 

  int Jopt=0;
  PetscOptionsGetInt(PETSC_NULL,"-Jopt",&Jopt,&flg);
  PetscPrintf(PETSC_COMM_WORLD,"----Jopt is: %d \n",Jopt);
  if(Jopt){

    VecSet(ptsrc,0.0);
    double *inJ;
    FILE *Jptf;
    inJ = (double *) malloc(6*Nr*Nz*sizeof(double));
    Jptf = fopen("Jinput.txt","r");
    int inJi;
    for (inJi=0;inJi<6*Nr*Nz;inJi++)
      {
	fscanf(Jptf,"%lf",&inJ[inJi]);
      }
    fclose(Jptf);

    ArrayToVec(inJ,ptsrc);

    free(inJ);

  }
  
  /*--------Create index sets for the vec scatter -------*/
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&from); CHKERRQ(ierr);
  ierr =ISCreateStride(PETSC_COMM_SELF,DegFree,0,1,&to); CHKERRQ(ierr);

  /*--------Setup the KSP variables ---------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the KSP variables.--------\n ");CHKERRQ(ierr);
  KSP ksp1a, ksp1b, ksp2;
  PC pc1a, pc1b, pc2; 
  int its1a=100, its1b=100, its2=100;
  int iteronly=0;
  setupKSP(PETSC_COMM_WORLD,&ksp1a,&pc1a,solver,iteronly,maxit);
  setupKSP(PETSC_COMM_WORLD,&ksp1b,&pc1b,solver,iteronly,maxit);
  setupKSP(PETSC_COMM_WORLD,&ksp2,&pc2,solver,iteronly,maxit);

  /*--------Setup Helmholtz filter---------*/
  PC pcH;
  GetH(PETSC_COMM_WORLD,&Hfilt,DegFree,1,1,sH,nR,dimH,&kspH,&pcH);
  PetscObjectSetName((PetscObject) Hfilt, "H");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Setting up the Hfilt DONE!--------\n ");CHKERRQ(ierr);

  /*---------Setup Done!---------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Everything set up! Ready to calculate the overlap and gradient.--------\n ");CHKERRQ(ierr);

  /*---------Setup the epsopt and grad arrays----------------*/
  double *epsopt;
  FILE *ptf;
  epsopt = (double *) malloc(DegFree*sizeof(double));
  ptf = fopen(initialdatafile,"r");
  PetscPrintf(PETSC_COMM_WORLD,"reading from input files \n");
  for (i=0;i<DegFree;i++)
    { 
      fscanf(ptf,"%lf",&epsopt[i]);
    }
  fclose(ptf);

  double *grad;
  grad = (double *) malloc(DegFree*sizeof(double));

  /**Set up J1a, J1aconj, b1a, x1a and LDOSdata1a common to all jobs**/
  Vec J1a, J1aconj, b1a, x1a;
  VecDuplicate(vR,&J1aconj);
  VecDuplicate(vR,&b1a);
  VecDuplicate(vR,&x1a);
  VecDuplicate(vR,&J1a);

  VecCopy(ptsrc,J1a);
  MatMult(C,J1a,J1aconj);
  
  MatMult(D,J1a,b1a);
  VecScale(b1a,omega1a);
  Vec ldos1agrad;
  VecDuplicate(epsSReal,&ldos1agrad);

  //LDOSdataGroup ldos1adata={omega1a,ksp1a,&its1a,M1a,b1a,x1a,J1aconj,epsSReal,epsFReal,epsDiff1a,epsMed1a,epscoef1a,ldos1agrad,outputbase,PETSC_NULL,0}; 

  /**Set up J1b, J1bconj, b1b, x1b and LDOSdata1b common to all jobs**/
  Vec J1b, J1bconj, b1b, x1b;
  VecDuplicate(vR,&J1bconj);
  VecDuplicate(vR,&b1b);
  VecDuplicate(vR,&x1b);
  VecDuplicate(vR,&J1b);

  VecCopy(ptsrc,J1b);
  MatMult(C,J1b,J1bconj);
  
  MatMult(D,J1b,b1b);
  VecScale(b1b,omega1b);
  Vec ldos1bgrad;
  VecDuplicate(epsSReal,&ldos1bgrad);

  //LDOSdataGroup ldos1bdata={omega1b,ksp1b,&its1b,M1b,b1b,x1b,J1bconj,epsSReal,epsFReal,epsDiff1b,epsMed1b,epscoef1b,ldos1bgrad,outputbase,PETSC_NULL,0}; 

  /***Set up done***/

  int Job;
  PetscOptionsGetInt(PETSC_NULL,"-Job",&Job,&flg); MyCheckAndOutputInt(flg,Job,"Job","Job");

if (Job==1){
  /*---------Optimization--------*/
  double mylb,myub, *lb=NULL, *ub=NULL;
  int maxeval, maxtime, mynloptalg;
  double maxf;
  nlopt_opt  opt;
  nlopt_result result;

  PetscOptionsGetInt(PETSC_NULL,"-maxeval",&maxeval,&flg);  MyCheckAndOutputInt(flg,maxeval,"maxeval","max number of evaluation");
  PetscOptionsGetInt(PETSC_NULL,"-maxtime",&maxtime,&flg);  MyCheckAndOutputInt(flg,maxtime,"maxtime","max time of evaluation");
  PetscOptionsGetInt(PETSC_NULL,"-mynloptalg",&mynloptalg,&flg);  MyCheckAndOutputInt(flg,mynloptalg,"mynloptalg","The algorithm used ");

  mylb=0, myub=1.0;      
 
  lb = (double *) malloc(DegFree*sizeof(double));
  ub = (double *) malloc(DegFree*sizeof(double));

  for(i=0;i<DegFree;i++)
  {
    lb[i] = mylb;
    ub[i] = myub;
  }

  int numfixedpxl=0;
  int fixedpxlstart=0;
  double valfixedpxl=1.0;
  PetscOptionsGetInt(PETSC_NULL,"-numfixedpxl",&numfixedpxl,&flg);
  if(!flg) numfixedpxl=0;
  PetscPrintf(PETSC_COMM_WORLD,"------numfixedpxl is %d \n",numfixedpxl);
  if(numfixedpxl){
    PetscOptionsGetInt(PETSC_NULL,"-fixedpxlstart",&fixedpxlstart,&flg); MyCheckAndOutputInt(flg,fixedpxlstart,"fixedpxlstart","start index of fixed pixel array");
    PetscOptionsGetReal(PETSC_NULL,"-valfixedpxl",&valfixedpxl,&flg); MyCheckAndOutputDouble(flg,valfixedpxl,"valfixedpxl","value of fixed pixels");
    for(i=fixedpxlstart;i<fixedpxlstart+numfixedpxl;i++){
      lb[i]=valfixedpxl;
      ub[i]=valfixedpxl;
    }
  }
  
  opt = nlopt_create(mynloptalg, DegFree);

  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,maxeval);
  nlopt_set_maxtime(opt,maxtime);
  if (mynloptalg==11)
  {
    nlopt_set_vector_storage(opt,4000);
  }

  int mynloptlocalalg;
  nlopt_opt local_opt;
  PetscOptionsGetInt(PETSC_NULL,"-mynloptlocalalg",&mynloptlocalalg,&flg);  MyCheckAndOutputInt(flg,mynloptlocalalg,"mynloptlocalalg","The local optimization algorithm used ");
  if (mynloptlocalalg)
  { 
	PetscPrintf(PETSC_COMM_WORLD,"-----------Running with a local optimizer.-----\n"); 
	local_opt=nlopt_create(mynloptlocalalg,DegFree);
	nlopt_set_ftol_rel(local_opt, 1e-14);
	nlopt_set_maxeval(local_opt,100000);
	nlopt_set_local_optimizer(opt,local_opt);
  }

  double ldospowerindex;
  PetscOptionsGetReal(PETSC_NULL,"-ldospowerindex",&ldospowerindex,&flg);  MyCheckAndOutputDouble(flg,ldospowerindex,"ldospowerindex","ldospowerindex");

  Vec ej,betagrad;
  VecDuplicate(vR,&ej);
  VecDuplicate(epsSReal,&betagrad);
  if(ptsrcdir==1) VecCopy(unitr,ej);
  if(ptsrcdir==2) VecCopy(unitp,ej);
  if(ptsrcdir==3) VecCopy(unitz,ej);

  SFGdataGroup sfgdata={ldospowerindex,omega1a,omega1b,omega2,ksp1a,ksp1b,ksp2,&its1a,&its1b,&its2,M1a,M1b,M2,b1a,b1b,x1a,x1b,ej,J1aconj,J1bconj,epsSReal,epsFReal,epsDiff1a,epsDiff1b,epsDiff2,epsMed1a,epsMed1b,epsMed2,epscoef1a,epscoef1b,epscoef2,ldos1agrad,ldos1bgrad,betagrad,outputbase,PETSC_NULL,PETSC_NULL};
  if(strcmp(vecNLfile,"")){
    sfgdata.vecNL=vecNL;
  }
  
  nlopt_set_max_objective(opt,optsfg,&sfgdata);
  
  result = nlopt_optimize(opt,epsopt,&maxf);

  if (result < 0) {
    PetscPrintf(PETSC_COMM_WORLD,"nlopt failed! \n", result);
  }
  else {
    PetscPrintf(PETSC_COMM_WORLD,"found extremum  %0.16e\n", maxf); 
  }

  PetscPrintf(PETSC_COMM_WORLD,"nlopt returned value is %d \n", result);

  VecDestroy(&ej);
  VecDestroy(&betagrad);
  nlopt_destroy(opt);

  int rankA;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rankA);

  if(rankA==0)
    {
      ptf = fopen(strcat(filenameComm,"epsopt.txt"),"w");
        for (i=0;i<DegFree;i++)
          fprintf(ptf,"%0.16e \n",epsopt[i]);
	  fclose(ptf);
    }

}

if (Job==2){

  /*---------Calculate the overlap and gradient--------*/
  int epsoptj;
  PetscOptionsGetInt(PETSC_NULL,"-epsoptj",&epsoptj,&flg);  MyCheckAndOutputInt(flg,epsoptj,"epsoptj","epsoptj");

  double beta;
  double s1=0.1, ds=0.01, s2=1.0+ds, epscen;
  
  double ldospowerindex;
  PetscOptionsGetReal(PETSC_NULL,"-ldospowerindex",&ldospowerindex,&flg);  MyCheckAndOutputDouble(flg,ldospowerindex,"ldospowerindex","ldospowerindex");

  Vec ej,betagrad;
  VecDuplicate(vR,&ej);
  VecDuplicate(epsSReal,&betagrad);
  if(ptsrcdir==1) VecCopy(unitr,ej);
  if(ptsrcdir==2) VecCopy(unitp,ej);
  if(ptsrcdir==3) VecCopy(unitz,ej);

  SFGdataGroup sfgdata={ldospowerindex,omega1a,omega1b,omega2,ksp1a,ksp1b,ksp2,&its1a,&its1b,&its2,M1a,M1b,M2,b1a,b1b,x1a,x1b,ej,J1aconj,J1bconj,epsSReal,epsFReal,epsDiff1a,epsDiff1b,epsDiff2,epsMed1a,epsMed1b,epsMed2,epscoef1a,epscoef1b,epscoef2,ldos1agrad,ldos1bgrad,betagrad,outputbase,PETSC_NULL,PETSC_NULL};
  if(strcmp(vecNLfile,"")){
    sfgdata.vecNL=vecNL;
  }
  
  for (epscen=s1;epscen<s2;epscen+=ds)
    {
      epsopt[epsoptj]=epscen;
      beta = optsfg(DegFree,epsopt,grad,&sfgdata);
      PetscPrintf(PETSC_COMM_WORLD,"epscen: %g beta: %g beta-grad: %g \n", epsopt[epsoptj], beta, grad[epsoptj]);
    }

  VecDestroy(&ej);
  VecDestroy(&betagrad);
    

 }

/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
/*------------------------------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD,"--------Done!--------\n ");CHKERRQ(ierr);

/* ----------------------Destroy Vecs and Mats----------------------------*/ 
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);
  ierr = MatDestroy(&C); CHKERRQ(ierr);
  ierr = MatDestroy(&D); CHKERRQ(ierr);
  ierr = MatDestroy(&M1a); CHKERRQ(ierr); 
  ierr = MatDestroy(&M1b); CHKERRQ(ierr);   
  ierr = MatDestroy(&M2); CHKERRQ(ierr);
  ierr = MatDestroy(&Hfilt); CHKERRQ(ierr);

  ierr = VecDestroy(&vR); CHKERRQ(ierr);
  ierr = VecDestroy(&vecQ); CHKERRQ(ierr);
  ierr = VecDestroy(&vecRad); CHKERRQ(ierr);
  ierr = VecDestroy(&weight); CHKERRQ(ierr);
  ierr = VecDestroy(&unitr); CHKERRQ(ierr);
  ierr = VecDestroy(&unitp); CHKERRQ(ierr);
  ierr = VecDestroy(&unitz); CHKERRQ(ierr);
  ierr = VecDestroy(&ptsrc); CHKERRQ(ierr);

  ierr = VecDestroy(&epsDiff1a); CHKERRQ(ierr);
  ierr = VecDestroy(&epsDiff1b); CHKERRQ(ierr);
  ierr = VecDestroy(&epsDiff2); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef1a); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef1b); CHKERRQ(ierr);
  ierr = VecDestroy(&epscoef2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsMed1a); CHKERRQ(ierr);
  ierr = VecDestroy(&epsMed1b); CHKERRQ(ierr);
  ierr = VecDestroy(&epsMed2); CHKERRQ(ierr);
  ierr = VecDestroy(&epsSReal); CHKERRQ(ierr);
  ierr = VecDestroy(&epsFReal); CHKERRQ(ierr);
  ierr = VecDestroy(&vgradlocal); CHKERRQ(ierr);

  ierr = VecDestroy(&x1a); CHKERRQ(ierr);
  ierr = VecDestroy(&b1a);CHKERRQ(ierr);
  ierr = VecDestroy(&J1a); CHKERRQ(ierr);
  ierr = VecDestroy(&J1aconj); CHKERRQ(ierr);
  ierr = VecDestroy(&ldos1agrad); CHKERRQ(ierr);
  ierr = VecDestroy(&x1b); CHKERRQ(ierr);
  ierr = VecDestroy(&b1b);CHKERRQ(ierr);
  ierr = VecDestroy(&J1b); CHKERRQ(ierr);
  ierr = VecDestroy(&J1bconj); CHKERRQ(ierr);
  ierr = VecDestroy(&ldos1bgrad); CHKERRQ(ierr);

  ierr = VecDestroy(&vecNL); CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp1a);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp1b);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp2);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspH);CHKERRQ(ierr);

  ISDestroy(&from);
  ISDestroy(&to);

  free(epsopt);
  free(grad);
  //free(lb);
  //free(ub);

  /*------------ finalize the program -------------*/

  {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Barrier(PETSC_COMM_WORLD);
  }
  
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}

