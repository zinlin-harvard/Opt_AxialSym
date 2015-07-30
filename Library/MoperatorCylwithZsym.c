#include <petsc.h>
#include "libOPT.h"
#include <complex.h>

/*

This rountine generates sparse matrix for the operator Curl \times 1/mu \times Curl:
		 
Output parameter: sparse matrix M.

 */

extern int zbl;

extern int Nr, Nz, Npmlr, Npmlz;
extern double hr, hz;
extern double Qabs;
extern int mPML;
extern double Refl;

#undef __FUNCT__ 
#define __FUNCT__ "MoperatorCyl"
PetscErrorCode MoperatorCyl(MPI_Comm comm, Mat *Aout, int m, double omega)
{
  Mat Dh,De,Muinv,Atmp,A;
  PetscErrorCode ierr;

  makeDh(comm,&Dh,m,omega);
  makeDe(comm,&De,m,omega);
  makeMuinv(comm,&Muinv);
     
  ierr = MatMatMult(Muinv,De,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Atmp); CHKERRQ(ierr);
  ierr = MatMatMult(Dh,Atmp,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A); CHKERRQ(ierr);
  MatScale(A,-1.0);

  ierr = PetscObjectSetName((PetscObject) Dh,  "Mh"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) De,  "Me"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) Muinv,  "Mu"); CHKERRQ(ierr);
  
  //OutputMat(PETSC_COMM_WORLD, Dh, "Dh",".m");
  //OutputMat(PETSC_COMM_WORLD, De, "De",".m");
  //OutputMat(PETSC_COMM_WORLD, Muinv, "Muinv",".m");

  MatDestroy(&De);
  MatDestroy(&Dh);
  MatDestroy(&Muinv);
  MatDestroy(&Atmp);

  *Aout = A;
  PetscFunctionReturn(0);
}

/**********Position manipulation functions****************/
void expandAddress(int Nz, int Nr, int Nc, int Nq, int i, gridInd *izrcq)
{
    int itmp;

    izrcq->iz = (itmp = i) % Nz;
    izrcq->ir = (itmp /= Nz) % Nr;
    izrcq->ic = (itmp /= Nr) % Nc;
    izrcq->iq = itmp / Nc;
}

int collapseAddress(int Nz, int Nr, int Nc, int Nq, gridInd *izrcq)
{
    int iztmp = izrcq->iz;
    int irtmp = izrcq->ir;
    int ictmp = izrcq->ic;
    int iqtmp = izrcq->iq;

    return iztmp + Nz * (irtmp + Nr * (ictmp + Nc * iqtmp));
}

/**********SC-PML factors****************/
double complex sz(double iz, double omega)
{
 double z  = iz*hz;
 double dz = (Npmlz+1e-10)*hz;
 double S  = (Npmlz!=0) * (-1.0)*(mPML + 1)*log(Refl)/(2.0*omega*dz);
 double ld = (Npmlz!=0) * (zbl==0) * (z<Npmlz*hz) * (Npmlz*hz-z)/dz + (Npmlz!=0) * (z>(Nz-Npmlz)*hz) * (z+Npmlz*hz-Nz*hz)/dz;
 
 double complex inv=1.0/(1.0 + I * S * pow(ld,mPML) );
 return inv;
}

double complex sr(double ir, double omega)
{
 double r  = ir*hr;
 double dr = (Npmlr+1e-10)*hr;
 double S  = (Npmlr!=0) * (-1.0)*(mPML + 1)*log(Refl)/(2.0*omega*dr);
 double ld = (Npmlr!=0) * (r>(Nr-Npmlr)*hr) * (r+Npmlr*hr-Nr*hr)/dr;
 
 double complex inv=1.0/(1.0 + I * S * pow(ld,mPML) );
 return inv;
}

/**********Create Dh, De and Muinv****************/

#undef __FUNCT__ 
#define __FUNCT__ "makeDh"
PetscErrorCode makeDh(MPI_Comm comm, Mat *Dhout, int m, double omega)
{

  Mat Dh;
  PetscErrorCode ierr;
  int Nc = 3, Nq = 2, Nzrcq=Nz*Nr*Nc*Nq;
  int ns, ne;
  int tmpiz,tmpir,tmpic,tmpiq;
  gridInd izrcq,j0zrcq,j1zrcq,j2zrcq,j3zrcq,j4zrcq,j5zrcq,j6zrcq,j7zrcq;
  int i,j[8];
  double val[8];
  double complex v1, v2, v3, v4;

  /*------Create Dh-----------*/

  MatCreate(comm, &Dh);
  MatSetType(Dh,MATMPIAIJ);
  MatSetSizes(Dh,PETSC_DECIDE, PETSC_DECIDE, Nzrcq, Nzrcq);
  MatMPIAIJSetPreallocation(Dh, 8, PETSC_NULL, 8, PETSC_NULL);

  ierr = MatGetOwnershipRange(Dh, &ns, &ne); CHKERRQ(ierr);
  
  for (i = ns; i < ne; ++i) {

  expandAddress(Nz,Nr,Nc,Nq,i,&izrcq);
  tmpiz=izrcq.iz;
  tmpir=izrcq.ir;
  tmpic=izrcq.ic;
  tmpiq=izrcq.iq;
  v1=0.0;v2=0.0;v3=0.0;v4=0.0;
  
  if(tmpic==0){
	
      v1 = 		   1.0*sz(tmpiz+0.5,omega)/hz;
      v2 = (tmpiz<Nz-1) ? -1.0*sz(tmpiz+0.5,omega)/hz : 0.0;
      v3 =		  -1.0*m/(hr*(tmpir+0.5));
      v4 = 		   0.0;

    };
  
  if(tmpic==1){ 

      v1 =		  -1.0*sz(tmpiz+0.5,omega)/hz;
      v2 = (tmpiz<Nz-1) ?  1.0*sz(tmpiz+0.5,omega)/hz : 0.0;
      v3 = (tmpir>0)    ? -1.0*sr(tmpir,omega)/hr : -2.0/hr*(m==1); 
      v4 = (tmpir>0)    ?  1.0*sr(tmpir,omega)/hr : 0.0; 

    };

  if(tmpic==2){ 

      v1 = (tmpir>0)    ?  1.0*m/(tmpir*hr) : 0.0;
      v2 = 		   0.0;
      v3 = (tmpir>0)    ?  1.0*(tmpir+0.5)*sr(tmpir,omega)/(tmpir*hr) : 4/hr*(m==0);
      v4 = (tmpir>0)    ? -1.0*(tmpir-0.5)*sr(tmpir,omega)/(tmpir*hr) : 0.0; 

    };  

  j0zrcq=(gridInd){tmpiz,tmpir,(tmpic==0),tmpiq};    
  j1zrcq=(gridInd){tmpiz,tmpir,(tmpic==0),(tmpiq+1)%Nq};

  j2zrcq=(gridInd){(tmpiz<Nz-1)?tmpiz+1:Nz-2,tmpir,(tmpic==0),tmpiq};
  j3zrcq=(gridInd){(tmpiz<Nz-1)?tmpiz+1:Nz-2,tmpir,(tmpic==0),(tmpiq+1)%Nq};

  j4zrcq=(gridInd){tmpiz,tmpir,(tmpic!=2)+1,tmpiq};
  j5zrcq=(gridInd){tmpiz,tmpir,(tmpic!=2)+1,(tmpiq+1)%Nq};

  j6zrcq=(gridInd){tmpiz,(tmpir>0)?tmpir-1:1,(tmpic!=2)+1,tmpiq};
  j7zrcq=(gridInd){tmpiz,(tmpir>0)?tmpir-1:1,(tmpic!=2)+1,(tmpiq+1)%Nq};

  j[0]=collapseAddress(Nz,Nr,Nc,Nq,&j0zrcq);
  j[1]=collapseAddress(Nz,Nr,Nc,Nq,&j1zrcq);
  j[2]=collapseAddress(Nz,Nr,Nc,Nq,&j2zrcq);
  j[3]=collapseAddress(Nz,Nr,Nc,Nq,&j3zrcq);
  j[4]=collapseAddress(Nz,Nr,Nc,Nq,&j4zrcq);
  j[5]=collapseAddress(Nz,Nr,Nc,Nq,&j5zrcq);
  j[6]=collapseAddress(Nz,Nr,Nc,Nq,&j6zrcq);
  j[7]=collapseAddress(Nz,Nr,Nc,Nq,&j7zrcq);

  val[0]=creal(v1); 
  val[1]=cimag(v1) * pow(-1.0,(tmpiq+1)%Nq);
  val[2]=creal(v2);
  val[3]=cimag(v2) * pow(-1.0,(tmpiq+1)%Nq);
  val[4]=creal(v3);
  val[5]=cimag(v3) * pow(-1.0,(tmpiq+1)%Nq);
  val[6]=creal(v4);
  val[7]=cimag(v4) * pow(-1.0,(tmpiq+1)%Nq);

  ierr = MatSetValues(Dh,1,&i,8,j,val,ADD_VALUES); CHKERRQ(ierr);  
    
  }

  ierr = MatAssemblyBegin(Dh, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Dh, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  *Dhout = Dh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "makeDe"
PetscErrorCode makeDe(MPI_Comm comm, Mat *Deout, int m, double omega)
{

  Mat De;
  PetscErrorCode ierr;
  int Nc = 3, Nq = 2, Nzrcq=Nz*Nr*Nc*Nq;
  int ns, ne;
  int tmpiz,tmpir,tmpic,tmpiq;
  gridInd izrcq,j0zrcq,j1zrcq,j2zrcq,j3zrcq,j4zrcq,j5zrcq,j6zrcq,j7zrcq;
  int i,j[8];
  double val[8];
  double complex v1, v2, v3, v4;

  /*------Create De-----------*/

  MatCreate(comm, &De);
  MatSetType(De,MATMPIAIJ);
  MatSetSizes(De,PETSC_DECIDE, PETSC_DECIDE, Nzrcq, Nzrcq);
  MatMPIAIJSetPreallocation(De, 8, PETSC_NULL, 8, PETSC_NULL);

  ierr = MatGetOwnershipRange(De, &ns, &ne); CHKERRQ(ierr);
  
  for (i = ns; i < ne; ++i) {

  expandAddress(Nz,Nr,Nc,Nq,i,&izrcq);
  tmpiz=izrcq.iz;
  tmpir=izrcq.ir;
  tmpic=izrcq.ic;
  tmpiq=izrcq.iq;
  v1=0.0;v2=0.0;v3=0.0;v4=0.0;
  
  if(tmpic==0){
	
      v1 = 		   1.0*sz(tmpiz,omega)/hz;
      v2 = (tmpiz>0)    ? -1.0*sz(tmpiz,omega)/hz : -zbl/hz;
      v3 = (tmpir>0)    ? -1.0*m/(hr*tmpir) : 0.0;
      v4 = (tmpir>0)    ?  0.0 : -1.0/hr*(m==1);

      v3 = v3 * (zbl!=1 || tmpiz>0);
      v4 = v4 * (zbl!=1 || tmpiz>0);

    };
  
  if(tmpic==1){ 

      v1 =		  -1.0*sz(tmpiz,omega)/hz;
      v2 = (tmpiz>0)    ?  1.0*sz(tmpiz,omega)/hz : zbl/hz;
      v3 = 		  -1.0*sr(tmpir+0.5,omega)/hr;
      v4 = (tmpir<Nr-1) ?  1.0*sr(tmpir+0.5,omega)/hr : 0.0; 

    };

  if(tmpic==2){ 

      v1 = 		   1.0*m/((tmpir+0.5)*hr);
      v2 = 		   0.0;
      v3 = 		   1.0* tmpir     *sr(tmpir+0.5,omega)/((tmpir+0.5)*hr);
      v4 = (tmpir<Nr-1) ? -1.0*(tmpir+1.0)*sr(tmpir+0.5,omega)/((tmpir+0.5)*hr) : 0.0; 

    };  

  j0zrcq=(gridInd){tmpiz,tmpir,(tmpic==0),tmpiq};    
  j1zrcq=(gridInd){tmpiz,tmpir,(tmpic==0),(tmpiq+1)%Nq};

  j2zrcq=(gridInd){(tmpiz>0)?tmpiz-1:0,tmpir,(tmpic==0),tmpiq};
  j3zrcq=(gridInd){(tmpiz>0)?tmpiz-1:0,tmpir,(tmpic==0),(tmpiq+1)%Nq};

  j4zrcq=(gridInd){tmpiz,tmpir,(tmpic!=2)+1,tmpiq};
  j5zrcq=(gridInd){tmpiz,tmpir,(tmpic!=2)+1,(tmpiq+1)%Nq};

  j6zrcq=(gridInd){tmpiz,(tmpir<Nr-1)?tmpir+1:Nr-2,(tmpic!=2)+1,tmpiq};
  j7zrcq=(gridInd){tmpiz,(tmpir<Nr-1)?tmpir+1:Nr-2,(tmpic!=2)+1,(tmpiq+1)%Nq};

  j[0]=collapseAddress(Nz,Nr,Nc,Nq,&j0zrcq);
  j[1]=collapseAddress(Nz,Nr,Nc,Nq,&j1zrcq);
  j[2]=collapseAddress(Nz,Nr,Nc,Nq,&j2zrcq);
  j[3]=collapseAddress(Nz,Nr,Nc,Nq,&j3zrcq);
  j[4]=collapseAddress(Nz,Nr,Nc,Nq,&j4zrcq);
  j[5]=collapseAddress(Nz,Nr,Nc,Nq,&j5zrcq);
  j[6]=collapseAddress(Nz,Nr,Nc,Nq,&j6zrcq);
  j[7]=collapseAddress(Nz,Nr,Nc,Nq,&j7zrcq);

  val[0]=creal(v1); 
  val[1]=cimag(v1) * pow(-1.0,(tmpiq+1)%Nq);
  val[2]=creal(v2);
  val[3]=cimag(v2) * pow(-1.0,(tmpiq+1)%Nq);
  val[4]=creal(v3);
  val[5]=cimag(v3) * pow(-1.0,(tmpiq+1)%Nq);
  val[6]=creal(v4);
  val[7]=cimag(v4) * pow(-1.0,(tmpiq+1)%Nq);

  ierr = MatSetValues(De,1,&i,8,j,val,ADD_VALUES); CHKERRQ(ierr);  
    
  }

  ierr = MatAssemblyBegin(De, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(De, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  *Deout = De;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "makeMuinv"
PetscErrorCode makeMuinv(MPI_Comm comm, Mat *Muinvout)
{

  Mat Muinv;
  PetscErrorCode ierr;
  int Nc = 3, Nq = 2, Nzrcq=Nz*Nr*Nc*Nq;
  int ns, ne;
  int tmpiz,tmpir,tmpic,tmpiq;
  gridInd izrcq,j0zrcq,j1zrcq;
  int i;
  int jmu[2];
  double muval[2];


  /*-----Create Muinv----------*/

  MatCreate(comm, &Muinv);
  MatSetType(Muinv,MATMPIAIJ);
  MatSetSizes(Muinv,PETSC_DECIDE, PETSC_DECIDE, Nzrcq, Nzrcq);
  MatMPIAIJSetPreallocation(Muinv, 8, PETSC_NULL, 8, PETSC_NULL);

  double mu=1.0;
  double complex muinvcx=1.0/(mu*(1.0 + I*1.0/Qabs));

  ierr = MatGetOwnershipRange(Muinv, &ns, &ne); CHKERRQ(ierr);
  
  for (i = ns; i < ne; ++i) {

  expandAddress(Nz,Nr,Nc,Nq,i,&izrcq);
  tmpiz=izrcq.iz;
  tmpir=izrcq.ir;
  tmpic=izrcq.ic;
  tmpiq=izrcq.iq;

  j0zrcq=(gridInd){tmpiz,tmpir,tmpic,tmpiq};
  j1zrcq=(gridInd){tmpiz,tmpir,tmpic,(tmpiq+1)%Nq};

  muval[0]=creal(muinvcx);
  muval[1]=cimag(muinvcx)*pow(-1.0,(tmpiq+1)%Nq);

  jmu[0]=collapseAddress(Nz,Nr,Nc,Nq,&j0zrcq);
  jmu[1]=collapseAddress(Nz,Nr,Nc,Nq,&j1zrcq);

  ierr = MatSetValues(Muinv,1,&i,2,jmu,muval,INSERT_VALUES); CHKERRQ(ierr);

  }
   
  ierr = MatAssemblyBegin(Muinv, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Muinv, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  *Muinvout = Muinv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetWeightVec"
PetscErrorCode GetWeightVec(Vec weight,int Nr, int Nz, int zbl)
{
  PetscErrorCode ierr;
  int i, j, ns, ne, ir, iz, ic;
  double value;

  int Nc = 3;
  ierr = VecGetOwnershipRange(weight,&ns,&ne); CHKERRQ(ierr);
   
  for(i=ns; i<ne; i++)
    {
      iz = (j = i) % Nz;
      ir = (j /= Nz) % Nr;
      ic = (j /= Nr) % Nc;       

      if(zbl==0){
	value = 1.0;
      }else if(zbl!=0){
	value = 2.0;
	if(ic==2 && iz==0) value = 1.0;
      }

      VecSetValue(weight, i, value, INSERT_VALUES);
    }
  ierr = VecAssemblyBegin(weight); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(weight); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
