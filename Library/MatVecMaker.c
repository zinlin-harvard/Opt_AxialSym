#include <stdio.h>
#include <math.h>
#include <petsc.h>

#undef __FUNCT__
#define __FUNCT__ "GetDotMat"
PetscErrorCode GetDotMat(MPI_Comm comm, Mat *Bout, int c1, int c2, int Nr, int Nz)
{
  int Nc=3, N=2*Nc*Nr*Nz;
  Mat B;
  int ns,ne,i,j,ir,iz,ic,iq;
  double value=1;
  int col;
  PetscErrorCode ierr;
  
  MatCreate(comm, &B);
  MatSetType(B,MATMPIAIJ);
  MatSetSizes(B,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(B, 1, PETSC_NULL, 1, PETSC_NULL);

  ierr = MatGetOwnershipRange(B, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    iz = (j = i) % Nz;
    ir = (j /= Nz) % Nr;
    ic = (j /= Nr) % Nc;
    iq =  j /= Nc;
    
    if(ic==c1){ 
      col = iz + Nz * (ir + Nr * (c2 + Nc * iq));   
      ierr = MatSetValue(B,i,col,value,INSERT_VALUES); CHKERRQ(ierr);
    }


  }

  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) B,"DotMatrix"); CHKERRQ(ierr);
  
  *Bout = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "ImagIMat"
PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N)
{
  Mat D;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &D); CHKERRQ(ierr); // here N is total length;
  
  MatCreate(comm, &D);
  MatSetType(D,MATMPIAIJ);
  MatSetSizes(D,PETSC_DECIDE, PETSC_DECIDE,N,N);
  MatMPIAIJSetPreallocation(D, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(D, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    int id = (i+N/2)%(N);
    double sign = pow(-1.0, (i<N/2));
    ierr = MatSetValue(D, i, id, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) D, "ImaginaryIMatrix"); CHKERRQ(ierr);
  
  *Dout = D;
  PetscFunctionReturn(0);
}



#undef __FUNCT__ 
#define __FUNCT__ "CongMat"
PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N)
{
  Mat C;
  int nz = 1; /* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;  
  int i;
     
  //ierr = MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N,N,nz, NULL, nz, NULL, &C); CHKERRQ(ierr);
  
  MatCreate(comm, &C);
  MatSetType(C,MATMPIAIJ);
  MatSetSizes(C,PETSC_DECIDE, PETSC_DECIDE, N, N);
  MatMPIAIJSetPreallocation(C, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(C, &ns, &ne); CHKERRQ(ierr);

  for (i = ns; i < ne; ++i) {
    double sign = pow(-1.0, (i>(N/2-1)));
    ierr = MatSetValue(C, i, i, sign, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) C,"CongMatrix"); CHKERRQ(ierr);
  
  *Cout = C;
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetRadiusVec"
PetscErrorCode GetRadiusVec(Vec vecRad, int Nr, int Nz, double hr, int m)
{
   PetscErrorCode ierr;
   int i, j, ir, ic, ns, ne;
   double value=1.0;
   ierr = VecGetOwnershipRange(vecRad,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       j = i;
       ir = (j /= Nz) % Nr;
       ic = (j /= Nr) % 3;
 
       if (ic==0) value = (ir + 0.5)*hr;
       if (ic==1) value = (ir + 0.0)*hr;
       if (ic==2) value = (ir + 0.0)*hr + (ir==0)*(m==0)*hr/8; 
        
       VecSetValue(vecRad, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(vecRad); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(vecRad); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetMediumVec"
PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub)
{
   PetscErrorCode ierr;
   int i, iz, ns, ne;
   double value;
   ierr = VecGetOwnershipRange(epsmedium,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       iz = i%Nz;
       if (iz<Mz)
	 value = epsair;
       else
	 value = epssub;
        
       VecSetValue(epsmedium, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(epsmedium); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(epsmedium); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetMediumVecwithSub"
PetscErrorCode GetMediumVecwithSub(Vec epsmedium,int Nr, int Nz, int Mr, int Mz, double epsair, double epssub, int Mzslab, int mr0, int mz0)
{
   PetscErrorCode ierr;
   int i, j, ir, iz, ns, ne;
   double value;
   ierr = VecGetOwnershipRange(epsmedium,&ns,&ne); CHKERRQ(ierr);
   for(i=ns;i<ne; i++)
     {
       iz = (j = i) % Nz;
       ir = (j /= Nz) % Nr;

       if (Mzslab==2){

         if (ir<mr0+Mr)
	   value = epssub;
         else
	   value = epsair;

       }else{

         if (iz<mz0)
	   value = epssub;
         else
	   value = epsair;

       }        

       VecSetValue(epsmedium, i, value, INSERT_VALUES);

       }

   ierr = VecAssemblyBegin(epsmedium); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(epsmedium); CHKERRQ(ierr);

   PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetRealPartVec"
PetscErrorCode GetRealPartVec(Vec vR, int N)
{
   PetscErrorCode ierr;
   int i, ns, ne;

   ierr = VecGetOwnershipRange(vR,&ns,&ne); CHKERRQ(ierr);

   for(i=ns; i<ne; i++)
     {
       if (i<N/2)
	 VecSetValue(vR,i,1.0,INSERT_VALUES);
       else
	 VecSetValue(vR,i,0.0,INSERT_VALUES);
     }

   ierr = VecAssemblyBegin(vR); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(vR); CHKERRQ(ierr);
   
   PetscFunctionReturn(0);
  
}

#undef __FUNCT__ 
#define __FUNCT__ "AddMuAbsorption"
PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add)
{
  //compute muinvpml/(1+i/Qabs)
  double Qinv = (add==0) ? 0.0: (1.0/Qabs);
  double d=1 + pow(Qinv,2);
  PetscErrorCode ierr;
  int N;
  ierr=VecGetSize(muinvpml,&N);CHKERRQ(ierr);

  double *ptmuinvpml;
  ierr=VecGetArray(muinvpml, &ptmuinvpml);CHKERRQ(ierr);

  int i;
  double a,b;
  for(i=0;i<N/2;i++)
    {
      a=ptmuinvpml[i];
      b=ptmuinvpml[i+N/2];      
      muinv[i]= (a+b*Qinv)/d;
      muinv[i+N/2]=(b-a*Qinv)/d;
    }
  ierr=VecRestoreArray(muinvpml,&ptmuinvpml);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "GetUnitVec"
PetscErrorCode GetUnitVec(Vec ej, int pol, int N)
{
   PetscErrorCode ierr;
   int i, j, ns, ne, ic;
   int Nc=3;
   int Nxyz=N/6;

   ierr = VecGetOwnershipRange(ej,&ns,&ne); CHKERRQ(ierr);

   for(i=ns; i<ne; i++)
     {
       j=i;
       ic = (j /= Nxyz) % Nc;
       
       if (ic==pol)
	 VecSetValue(ej,i,1.0,INSERT_VALUES);
       else
	 VecSetValue(ej,i,0.0,INSERT_VALUES);
     }

   ierr = VecAssemblyBegin(ej); CHKERRQ(ierr);
   ierr = VecAssemblyEnd(ej); CHKERRQ(ierr);
   
   PetscFunctionReturn(0);
  
}

#undef __FUNCT__ 
#define __FUNCT__ "myinterp"
PetscErrorCode myinterp(MPI_Comm comm, Mat *Aout, int Nr, int Nz, int Mr, int Mz, int mr, int mz, int Mzslab)
{

  //(mr,mz) specifies the origin of the DOF region.
  //DOF region is a rectangle spanning diagonally from (mr,mz) to (mr+Mr-1,mz+Mz-1).
  //We keep the z invariant.

  Mat A;
  int nz = 1; 			/* max # nonzero elements in each row */
  PetscErrorCode ierr;
  int ns, ne;
  int i;

  int DegFree= (Mzslab==0)*Mr*Mz + (Mzslab==1)*Mr + (Mzslab==2)*Mz; 
 
  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, 6*Nr*Nz, DegFree);
  MatMPIAIJSetPreallocation(A, nz, PETSC_NULL, nz, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  double shift=0.5;
  for (i = ns; i < ne; ++i) {
    int ir, iz, ic;
    double rd, zd;
    int ird, izd;
    int j, id;

    iz = (j = i) % Nz;
    ir = (j /= Nz) % Nr;
    ic = (j /= Nr) % 3;

    rd = (ir-mr) + (ic!= 0)*shift;
    ird = ceil(rd-0.5);
    if (ird < 0 || ird >= Mr) continue;

    zd = (iz-mz) + (ic!= 2)*shift;
    izd = ceil(zd-0.5);   
    if (izd < 0 || izd >= Mz) continue;

    if (Mzslab==1) {
      id = ird;
    }else if (Mzslab==2){
      id = izd;
    }else{
      id = izd + Mz * ird;  
    }

    ierr = MatSetValue(A, i, id, 1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) A, "InterpMatrix"); CHKERRQ(ierr);
  *Aout = A;
  PetscFunctionReturn(0);
}

PetscErrorCode expandMat(MPI_Comm comm, Mat *Aout, int DegFree, int multiplier)
{
  PetscErrorCode ierr;
  int i,j,ns,ne;
  Mat A;
  
  MatCreate(comm, &A);
  MatSetType(A,MATMPIAIJ);
  MatSetSizes(A,PETSC_DECIDE, PETSC_DECIDE, multiplier*DegFree, DegFree);
  MatMPIAIJSetPreallocation(A, 1, PETSC_NULL, 1, PETSC_NULL);

  ierr = MatGetOwnershipRange(A, &ns, &ne); CHKERRQ(ierr);

  for(i=ns;i<ne;i++){

    j=i/multiplier;
    ierr = MatSetValue(A,i,j,1.0,INSERT_VALUES); CHKERRQ(ierr);

  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  *Aout = A;
  PetscFunctionReturn(0);

}

PetscErrorCode myinterpmultiplier(MPI_Comm comm, Mat *Aout, int Nr, int Nz, int multiplier, int Mr, int Mz, int mr, int mz, int Mzslab)
{

  PetscErrorCode ierr;
  Mat A,A1,A2;

  if(Mzslab==1){
    myinterp(comm,&A1,Nr,Nz,multiplier*Mr,Mz,mr,mz,Mzslab);
  }else if(Mzslab==2){
    myinterp(comm,&A1,Nr,Nz,Mr,multiplier*Mz,mr,mz,Mzslab);
  }   else{
    PetscPrintf(comm,"!!!!!you cannot use myinterpmultiplier with Mzslab=0. So defaulted to Mzslab=1!!!!\n");
    myinterp(comm,&A1,Nr,Nz,multiplier*Mr,Mz,mr,mz,1);
  }

  int DegFree=(Mzslab==1)? Mr : Mz;
  expandMat(comm,&A2,DegFree,multiplier);

  ierr = MatMatMult(A1,A2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A); CHKERRQ(ierr);

  MatDestroy(&A1);
  MatDestroy(&A2);
  
  *Aout = A;
  PetscFunctionReturn(0); 

}
