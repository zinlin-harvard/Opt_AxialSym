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
#define __FUNCT__ "computebeta2"
double computebeta2(Vec x1, Vec ej, int *its, KSP ksp1, KSP ksp2, Mat Mone, Mat Mtwo, double omega1, double omega2, Vec epsFReal, Vec epscoef1, Vec epscoef2, Vec betagrad)
{
  
  PetscErrorCode ierr;
  double hzr=hr*hz;
  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Second Harmonic Power and derivative------ \n");

  Vec x1j, J2, J2conj, b2, x2;
  VecDuplicate(vR,&x1j);
  VecDuplicate(vR,&J2);
  VecDuplicate(vR,&J2conj);
  VecDuplicate(vR,&b2);
  VecDuplicate(vR,&x2);

  VecPointwiseMult(x1j,x1,ej);
  CmpVecProd(x1j,x1j,J2);
  VecPointwiseMult(J2,J2,epsFReal);
  VecPointwiseMult(J2,J2,ej);
  MatMult(C,J2,J2conj);
  MatMult(D,J2,b2);
  VecScale(b2,omega2);

  SolveMatrix(PETSC_COMM_WORLD,ksp2,Mtwo,b2,x2,its);
   
  /*-------------Calculate and print out the beta2----------*/
  double betar, betai, beta;
  Vec tmp;
  VecDuplicate(vR,&tmp);
  VecPointwiseMult(tmp,J2conj,vecRad);
  VecPointwiseMult(tmp,tmp,weight);
  CmpVecDot(tmp,x2,&betar,&betai);
  beta=-1.0*hzr*betar;
  PetscPrintf(PETSC_COMM_WORLD,"---*****The current beta2 at frequency %g and step %.5d is %.16e \n",omega2/(2.0*PI),count,beta);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double t1,t2,tpast;
  //Calculate the derivative
  Vec Uone, Utwo, Uthree, u1, u2, u3, Grad0, Grad1, Grad2, Grad3, Grad4, x1jsq;
  VecDuplicate(vR,&Uone);
  VecDuplicate(vR,&Utwo);
  VecDuplicate(vR,&Uthree);
  VecDuplicate(vR,&u1);
  VecDuplicate(vR,&u2);
  VecDuplicate(vR,&u3);
  VecDuplicate(vR,&Grad0);
  VecDuplicate(vR,&Grad1);
  VecDuplicate(vR,&Grad2);
  VecDuplicate(vR,&Grad3);
  VecDuplicate(vR,&Grad4);
  VecDuplicate(vR,&x1jsq);

  //Uone = eps * 2 * x1j * conj(x2) * ej;
  ierr = MatMult(C,x2,tmp); CHKERRQ(ierr);
  CmpVecProd(tmp,x1j,Uone);
  ierr = VecPointwiseMult(Uone,Uone,ej); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Uone,Uone,epsFReal); CHKERRQ(ierr);
  VecScale(Uone,2.0);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1,Uone,u1);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone * u1 = Uone is %f s \n",tpast);

  //Utwo = eps * conj(x1j)^2 * ej = J2conj;
  VecCopy(J2conj,Utwo);
  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp2,Utwo,u2);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mtwo * u2 = Utwo is %f s \n",tpast);

  //Uthree = 2 * eps * x1j * u2 * ej;
  CmpVecProd(x1j,u2,Uthree);
  ierr = VecPointwiseMult(Uthree,Uthree,epsFReal); CHKERRQ(ierr);
  ierr = VecPointwiseMult(Uthree,Uthree,ej); CHKERRQ(ierr);
  VecScale(Uthree,2.0);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1,Uthree,u3);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mone * u3 = Uthree is %f s \n",tpast);

  //Grad0 = conj(x1j)^2 * ej * x2;
  CmpVecProd(x1j,x1j,x1jsq);
  MatMult(C,x1jsq,tmp);
  CmpVecProd(tmp,x2,Grad0);
  VecPointwiseMult(Grad0,Grad0,ej);

  //Grad1 = conj( u1 epscoef1 x1 );
  CmpVecProd(epscoef1,x1,Grad1);
  CmpVecProd(Grad1,u1,tmp);
  MatMult(C,tmp,Grad1);

  //Grad2 = u2 * epscoef2 * x2;
  CmpVecProd(epscoef2,x2,tmp);
  CmpVecProd(tmp,u2,Grad2);

  //Grad3 = i*omega2 * x1j^2 * u2 * ej;
  CmpVecProd(x1jsq,u2,tmp);
  MatMult(D,tmp,Grad3);
  VecPointwiseMult(Grad3,Grad3,ej);
  VecScale(Grad3,omega2);

  //Grad4 = i*omega2 * u3 * epscoef1 * x1;
  CmpVecProd(x1,epscoef1,Grad4);
  CmpVecProd(Grad4,u3,tmp);
  MatMult(D,tmp,Grad4);
  VecScale(Grad4,omega2);
     
  VecSet(tmp,0.0);
  VecAXPY(tmp,1.0,Grad0);
  VecAXPY(tmp,1.0,Grad1);
  VecAXPY(tmp,1.0,Grad2);
  VecAXPY(tmp,1.0,Grad3);
  VecAXPY(tmp,1.0,Grad4);

  VecPointwiseMult(tmp,tmp,vR);
  VecPointwiseMult(tmp,tmp,vecRad);
  VecPointwiseMult(tmp,tmp,weight);
  VecScale(tmp,-hzr);

  MatMultTranspose(A,tmp,betagrad);

  VecDestroy(&x1j);
  VecDestroy(&J2);
  VecDestroy(&J2conj);
  VecDestroy(&b2);
  VecDestroy(&x2);
  VecDestroy(&tmp);
  VecDestroy(&Uone);
  VecDestroy(&Utwo);
  VecDestroy(&Uthree);
  VecDestroy(&u1);
  VecDestroy(&u2);
  VecDestroy(&u3);
  VecDestroy(&Grad0);
  VecDestroy(&Grad1);
  VecDestroy(&Grad2);
  VecDestroy(&Grad3);
  VecDestroy(&Grad4);
  VecDestroy(&x1jsq);

  return beta;
}

