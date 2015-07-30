#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <petsc.h>
#include "libOPT.h"

double hvtanh(double x, double k, double x0);

double test(int DegFree, double *epsopt, double *grad, void *data)
{
  double sum;
  int i;
  sum=0;
  for(i=0;i<DegFree;i++){
    sum += epsopt[i]*epsopt[i];
    grad[i]= 2.0*epsopt[i];
  }
  PetscPrintf(PETSC_COMM_WORLD,"-----obj func is: %g \n",sum);

  return sum;

}

double hvtanh(double x, double k, double x0)
{
  return (0.5*tanh(k*x - k*x0)+0.5);
}

double lvs1d_opt(double (*optfunc)(int,double*,double*,void*), int DegFree, double *epsopt, double *grad, void *data, int maxeval, double dt, int steplength)
{
  //double epsopt to int struc
  int *struc;
  int i,j;
  double obj=0;
  struc=(int *) malloc(DegFree*sizeof(int));
  for(i=0;i<DegFree;i++){
    struc[i]=round(epsopt[i]);
    epsopt[i]=(double)struc[i];
  }
  
  for(i=0;i<maxeval;i++){
    obj=optfunc(DegFree,epsopt,grad,data);
    update_struc1d(struc,grad,DegFree,dt,steplength);
    for(j=0;j<DegFree;j++){
      epsopt[j]=(double)struc[j];
    }
  }
  
  free(struc);
  return obj;

}

void update_struc1d(int *struc, double *v, int n, double dt, int steplength)
{

  double Dt=dt/find_maxabs(v,n);
  PetscPrintf(PETSC_COMM_WORLD,"---Dtdebug: %g \n",Dt); //DEBUG
  int i,j,k,nstep;
  double dp, dm, u1, u2;
  int *strucfull;
  double *vfull, *lsf, *tmplsf;
  
  //struc to lsf (use strucfull)
  strucfull=(int *) malloc((n+2)*sizeof(int));
  vfull=(double *) malloc((n+2)*sizeof(double));
  lsf=(double *) malloc((n+2)*sizeof(double));
  tmplsf=(double *) malloc((n+2)*sizeof(double));
  for(i=1;i<n+1;i++){
    strucfull[i]=struc[i-1];
    vfull[i]=v[i-1];
    // vfull[i]=vfull[i]*(-1.0); //possible change here
  }
  strucfull[0]=0;
  strucfull[n+1]=0;
  vfull[0]=0;
  vfull[n+1]=0;

  struc2lsf_1d(strucfull,lsf,n+2);
    
  //lsf update using upwind difference scheme
  for(nstep=0;nstep<steplength;nstep++){
    for(j=0;j<n+2;j++){
      i=(j==0)? (n+1) : (j-1);
      k=(j>n) ?   0   : (j+1);
      dp=lsf[k]-lsf[j];
      dm=lsf[j]-lsf[i];
      u1=sqrt( min(dm,0)*min(dm,0) + max(dp,0)*max(dp,0) );
      u2=sqrt( max(dm,0)*max(dm,0) + min(dp,0)*min(dp,0) );
      //u1=fabs(dp);
      //u2=fabs(dm);
      tmplsf[j] = lsf[j] - Dt * min(vfull[j],0) * u1 - Dt * max(vfull[j],0) * u2;
    }

    for(j=0;j<n+2;j++){
      lsf[j]=tmplsf[j];
    }

    //DEBUG
    for(j=0;j<n+2;j++){
      PetscPrintf(PETSC_COMM_WORLD,"----lsfdebug: %g \n",lsf[j]);
    }
    PetscPrintf(PETSC_COMM_WORLD,"----lsfdebug:   \n");
    //DEBUG
    
  }
  //tmplsf to struc (clip the sturcfull to struc)
  lsf2struc_1d(lsf,strucfull,n+2);
  for(i=0;i<n;i++){
    struc[i]=strucfull[i+1];
  }
  
  free(strucfull);
  free(vfull);
  free(lsf);
  free(tmplsf);
  
}

void struc2lsf_1d(int *struc, double *lsf, int n)
{

  int *struc_neg;
  double *dist, *dist_neg;
  int i;
  dist=(double *) malloc(n*sizeof(double));
  struc_neg=(int *) malloc(n*sizeof(int));
  dist_neg=(double *) malloc(n*sizeof(double));

  edtrans1d(struc,dist,n);
  
  for(i=0;i<n;i++){
    if(struc[i]==0)
      struc_neg[i]=1;
    else
      struc_neg[i]=0;
  }
  edtrans1d(struc_neg,dist_neg,n);

  for(i=0;i<n;i++){
    lsf[i] = struc_neg[i] * ( dist[i] - 0.5 ) - struc[i] * ( dist_neg[i] - 0.5 );
  }

  free(dist);
  free(struc_neg);
  free(dist_neg);

}

void lsf2struc_1d(double *lsf, int *struc, int n)
{

  int i;
  for(i=0;i<n;i++){
    if(lsf[i]<0)
      struc[i]=1;
    else
      struc[i]=0;
  }

}

void edtrans1d(int *struc, double *dist, int n)
{

  int i,j;
  double *posj;
  double tmpdist;
  posj=(double *) malloc(n*sizeof(double));
  for(i=0;i<n;i++){
    if(struc[i]==1)
      posj[i]=i;
    else
      posj[i]=-(n+1);
  }

  for(i=0;i<n;i++){
    if(struc[i]==1){
      dist[i]=0;}
    else{
      dist[i]=n+1;
      for(j=0;j<n;j++){
	tmpdist=fabs(i-posj[j]);
	if(tmpdist<dist[i]) dist[i]=tmpdist;
      }
    }
  }

  free(posj);

}

double find_maxabs(double *v, int n)
{

  int i;
  double maxval;
  maxval=fabs(v[0]);
  for(i=0;i<n;i++){
    if(maxval<fabs(v[i])) maxval=fabs(v[i]);
  }

  return maxval;
}

double min(double a, double b)
{
  return ( (a<b) ? a : b );
}

double max(double a, double b)
{
  return ( (a>b) ? a : b);
}
