#include <petsc.h>
#include <time.h>
#include "libOPT.h"
#include <complex.h>
#include "petsctime.h"

#define Ptime PetscTime

extern int Nr, Nz;
extern double hr, hz;
extern Mat A,C,D;
extern Vec vR, vecRad, vecQ, weight;

#undef __FUNCT__ 
#define __FUNCT__ "funcWdotEabs"
double funcWdotEabs(KSP ksp, Vec W, Vec epscoef, Vec grad, double omega)
{
  Vec x, rhs;
  VecDuplicate(vR,&x);
  VecDuplicate(vR,&rhs);
  PetscErrorCode ierr;
  double hzr=hr*hz;
  PetscPrintf(PETSC_COMM_WORLD,"----Calculating WdotE.------- \n");

  MatMult(D,W,rhs);
  VecScale(rhs,omega);
  KSPSolve(ksp,rhs,x);
  
  Vec tmp, Grad;
  VecDuplicate(vR,&tmp);
  VecDuplicate(vR,&Grad);

  /*-------------Calculate and print out the LDOS at the given freq----------*/
  double output;
  CmpVecProd(x,W,tmp);
  VecPointwiseMult(tmp,tmp,vecRad);
  VecPointwiseMult(tmp,tmp,weight);
  VecPointwiseMult(tmp,tmp,vR);
  VecScale(tmp,-1.0*hzr);
  VecSum(tmp,&output);
  PetscPrintf(PETSC_COMM_WORLD,"****WdotE at this step is: %g \n", output);
  
  //gradient = x * x * epscoef * (-I*1/omega)
  
  CmpVecProd(x,x,Grad);
  CmpVecProd(Grad,epscoef,tmp);
  MatMult(D,tmp,Grad);
  VecScale(Grad,-1.0/omega);
  VecPointwiseMult(Grad,Grad,vecRad);
  VecPointwiseMult(Grad,Grad,weight);
  VecPointwiseMult(Grad,Grad,vR);
  VecScale(Grad,-1.0*hzr);

  MatMultTranspose(A,Grad,grad);

  ierr=VecDestroy(&x);CHKERRQ(ierr);
  ierr=VecDestroy(&rhs);CHKERRQ(ierr);
  ierr=VecDestroy(&tmp);CHKERRQ(ierr);
  ierr=VecDestroy(&Grad);CHKERRQ(ierr);

  return output;
}

