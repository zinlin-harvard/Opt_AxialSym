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
extern Vec vR, vecRad, vecQ, weight;

#undef __FUNCT__ 
#define __FUNCT__ "computeldos"
double computeldos(KSP ksp, Mat Mopr, double omega, Vec epsFReal, Vec b, Vec Jconj, Vec x, Vec epscoef,Vec ldosgrad, int *its)
{
  
  PetscErrorCode ierr;
  double hzr=hr*hz;
  PetscPrintf(PETSC_COMM_WORLD,"----Calculating LDOS and derivative.------- \n");

  Vec tmp, Grad;
  VecDuplicate(vR,&tmp);
  VecDuplicate(vR,&Grad);

  SolveMatrix(PETSC_COMM_WORLD,ksp,Mopr,b,x,its);
   
  /*-------------Calculate and print out the LDOS at the given freq----------*/
  //ldos = -Re(conj(J)'*E) 
  double ldosr, ldosi, ldos;
  VecPointwiseMult(tmp,x,vecRad);
  VecPointwiseMult(tmp,tmp,weight);
  CmpVecDot(tmp,Jconj,&ldosr,&ldosi);
  ldos=-1.0*hzr*ldosr;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current ldos at frequency %g and step %.5d is %.16e \n",omega/(2.0*PI),count,ldos);

  //gradient_ldos = hzr * Re[ x^2 epscoef wt Rad I/omega ]
    CmpVecProd(x,x,Grad);
    CmpVecProd(Grad,epscoef,tmp);
    MatMult(D,tmp,Grad);
    VecPointwiseMult(Grad,Grad,vecRad);
    VecPointwiseMult(Grad,Grad,weight);
    VecScale(Grad,hzr/omega);
    VecPointwiseMult(Grad,Grad,vR);

    MatMultTranspose(A,Grad,ldosgrad);

    ierr=VecDestroy(&tmp);CHKERRQ(ierr);
    ierr=VecDestroy(&Grad);CHKERRQ(ierr);

  return ldos;
}

