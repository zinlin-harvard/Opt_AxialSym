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
#define __FUNCT__ "computesfg"
double computesfg(Vec x1a, Vec x1b, Vec x2, Vec ej, int *its, KSP ksp1a, KSP ksp1b, KSP ksp2, Mat MoneA, Mat MoneB, Mat Mtwo, double omega2, Vec epsFReal, Vec epscoef1a, Vec epscoef1b, Vec epscoef2, Vec betagrad, Vec vecNL)
{
  
  PetscErrorCode ierr;
  double hzr=hr*hz;
  PetscPrintf(PETSC_COMM_WORLD,"----Calculating Sum Frequency Power and derivative------ \n");

  Vec x1aj, x1bj, J2, J2conj, b2;
  VecDuplicate(vR,&x1aj);
  VecDuplicate(vR,&x1bj);
  VecDuplicate(vR,&J2);
  VecDuplicate(vR,&J2conj);
  VecDuplicate(vR,&b2);

  VecPointwiseMult(x1aj,x1a,ej);
  VecPointwiseMult(x1bj,x1b,ej);
  CmpVecProd(x1aj,x1bj,J2);
  if(vecNL)
    VecPointwiseMult(J2,J2,vecNL);
  else
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
  Vec UoneA, UoneB, Utwo, UthreeA, UthreeB, u1a, u1b, u2, u3a, u3b, Grad0, Grad1a, Grad1b, Grad2, Grad3, Grad4a, Grad4b, xabjsq;
  VecDuplicate(vR,&UoneA);
  VecDuplicate(vR,&UoneB);
  VecDuplicate(vR,&Utwo);
  VecDuplicate(vR,&UthreeA);
  VecDuplicate(vR,&UthreeB);
  VecDuplicate(vR,&u1a);
  VecDuplicate(vR,&u1b);
  VecDuplicate(vR,&u2);
  VecDuplicate(vR,&u3a);
  VecDuplicate(vR,&u3b);
  VecDuplicate(vR,&Grad0);
  VecDuplicate(vR,&Grad1a);
  VecDuplicate(vR,&Grad1b);
  VecDuplicate(vR,&Grad2);
  VecDuplicate(vR,&Grad3);
  VecDuplicate(vR,&Grad4a);
  VecDuplicate(vR,&Grad4b);
  VecDuplicate(vR,&xabjsq);

  //UoneA = eps * x1bj * conj(x2) * ej;
  ierr = MatMult(C,x2,tmp); CHKERRQ(ierr);
  CmpVecProd(tmp,x1bj,UoneA);
  ierr = VecPointwiseMult(UoneA,UoneA,ej); CHKERRQ(ierr);
  if (vecNL)
    ierr = VecPointwiseMult(UoneA,UoneA,vecNL);
  else
    ierr = VecPointwiseMult(UoneA,UoneA,epsFReal);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1a,UoneA,u1a);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving MoneA * u1a = UoneA is %f s \n",tpast);

  //UoneB = eps * x1aj * conj(x2) * ej;
  ierr = MatMult(C,x2,tmp); CHKERRQ(ierr);
  CmpVecProd(tmp,x1aj,UoneB);
  ierr = VecPointwiseMult(UoneB,UoneB,ej); CHKERRQ(ierr);
  if (vecNL)
    ierr = VecPointwiseMult(UoneB,UoneB,vecNL);
  else
    ierr = VecPointwiseMult(UoneB,UoneB,epsFReal);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1b,UoneB,u1b);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving MoneB * u1b = UoneB is %f s \n",tpast);

  //Utwo = J2conj;
  VecCopy(J2conj,Utwo);
  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp2,Utwo,u2);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving Mtwo * u2 = Utwo is %f s \n",tpast);

  //UthreeA = eps * x1bj * u2 * ej;
  CmpVecProd(x1bj,u2,UthreeA);
  if(vecNL)
    ierr = VecPointwiseMult(UthreeA,UthreeA,vecNL); 
  else
    ierr = VecPointwiseMult(UthreeA,UthreeA,epsFReal);
  ierr = VecPointwiseMult(UthreeA,UthreeA,ej); CHKERRQ(ierr);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1a,UthreeA,u3a);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving MoneA * u3a = UthreeA is %f s \n",tpast);

  //UthreeB = eps * x1aj * u2 * ej;
  CmpVecProd(x1aj,u2,UthreeB);
  if(vecNL)
    ierr = VecPointwiseMult(UthreeB,UthreeB,vecNL); 
  else
    ierr = VecPointwiseMult(UthreeB,UthreeB,epsFReal);
  ierr = VecPointwiseMult(UthreeB,UthreeB,ej); CHKERRQ(ierr);

  ierr = Ptime(&t1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp1b,UthreeB,u3b);CHKERRQ(ierr);
  ierr = Ptime(&t2);CHKERRQ(ierr);
  tpast=t2-t1;
  if(rank==0)
    PetscPrintf(PETSC_COMM_SELF,"---The runing time for solving MoneB * u3b = UthreeB is %f s \n",tpast);

  //Grad0 = conj(x1aj * x1bj) * ej * x2;
  CmpVecProd(x1aj,x1bj,xabjsq);
  MatMult(C,xabjsq,tmp);
  CmpVecProd(tmp,x2,Grad0);
  VecPointwiseMult(Grad0,Grad0,ej);
  if(vecNL) VecScale(Grad0,0);
  
  //Grad1a = conj( u1a epscoef1a x1a );
  CmpVecProd(epscoef1a,x1a,Grad1a);
  CmpVecProd(Grad1a,u1a,tmp);
  MatMult(C,tmp,Grad1a);

  //Grad1b = conj( u1b epscoef1b x1b );
  CmpVecProd(epscoef1b,x1b,Grad1b);
  CmpVecProd(Grad1b,u1b,tmp);
  MatMult(C,tmp,Grad1b);

  //Grad2 = u2 * epscoef2 * x2;
  CmpVecProd(epscoef2,x2,tmp);
  CmpVecProd(tmp,u2,Grad2);

  //Grad3 = i*omega2 * x1aj * x1bj * u2 * ej;
  CmpVecProd(xabjsq,u2,tmp);
  MatMult(D,tmp,Grad3);
  VecPointwiseMult(Grad3,Grad3,ej);
  VecScale(Grad3,omega2);
  if(vecNL) VecScale(Grad3,0);
  
  //Grad4a = i*omega2 * u3a * epscoef1a * x1a;
  CmpVecProd(x1a,epscoef1a,Grad4a);
  CmpVecProd(Grad4a,u3a,tmp);
  MatMult(D,tmp,Grad4a);
  VecScale(Grad4a,omega2);

  //Grad4b = i*omega2 * u3b * epscoef1b * x1b;
  CmpVecProd(x1b,epscoef1b,Grad4b);
  CmpVecProd(Grad4b,u3b,tmp);
  MatMult(D,tmp,Grad4b);
  VecScale(Grad4b,omega2);

     
  VecSet(tmp,0.0);
  VecAXPY(tmp,1.0,Grad0);
  VecAXPY(tmp,1.0,Grad1a);
  VecAXPY(tmp,1.0,Grad1b);
  VecAXPY(tmp,1.0,Grad2);
  VecAXPY(tmp,1.0,Grad3);
  VecAXPY(tmp,1.0,Grad4a);
  VecAXPY(tmp,1.0,Grad4b);

  VecPointwiseMult(tmp,tmp,vR);
  VecPointwiseMult(tmp,tmp,vecRad);
  VecPointwiseMult(tmp,tmp,weight);
  VecScale(tmp,-hzr);

  MatMultTranspose(A,tmp,betagrad);

  VecDestroy(&x1aj);
  VecDestroy(&x1bj);
  VecDestroy(&J2);
  VecDestroy(&J2conj);
  VecDestroy(&b2);
  VecDestroy(&tmp);
  VecDestroy(&UoneA);
  VecDestroy(&UoneB);
  VecDestroy(&Utwo);
  VecDestroy(&UthreeA);
  VecDestroy(&UthreeB);
  VecDestroy(&u1a);
  VecDestroy(&u1b);
  VecDestroy(&u2);
  VecDestroy(&u3a);
  VecDestroy(&u3b);
  VecDestroy(&Grad0);
  VecDestroy(&Grad1a);
  VecDestroy(&Grad1b);
  VecDestroy(&Grad2);
  VecDestroy(&Grad3);
  VecDestroy(&Grad4a);
  VecDestroy(&Grad4b);
  VecDestroy(&xabjsq);

  return beta;
}

