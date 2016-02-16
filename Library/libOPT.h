#include <complex.h>

//define macros;
#undef PI
#define PI 3.14159265358979e+00

typedef struct{
  double omega;
  KSP ksp;
  int *its;
  Mat M;
  Vec b;
  Vec x;
  Vec Jconj;
  Vec epsSReal;
  Vec epsFReal;
  Vec epsDiff;
  Vec epsMed;
  Vec epscoef;
  Vec ldosgrad;
  int outputbase;
  Vec W;
  double expW;
} LDOSdataGroup;

typedef struct{
  double ldospowerindex;
  double omega1;
  double omega2;
  KSP ksp1;
  KSP ksp2;
  int *its1;
  int *its2;
  Mat M1;
  Mat M2;
  Vec b1;
  Vec x1;
  Vec ej;
  Vec J1conj;
  Vec epsSReal;
  Vec epsFReal;
  Vec epsDiff1;
  Vec epsDiff2;
  Vec epsMed1;
  Vec epsMed2;
  Vec epscoef1;
  Vec epscoef2;
  Vec ldos1grad;
  Vec betagrad;
  int outputbase;
  Mat B;
  Vec W;
  double expW;
  Vec vecNL;
} SHGdataGroup;

typedef struct{
  double ldospowerindex;
  double omega1a;
  double omega1b;
  double omega2;
  KSP ksp1a;
  KSP ksp1b;
  KSP ksp2;
  int *its1a;
  int *its1b;
  int *its2;
  Mat M1a;
  Mat M1b;
  Mat M2;
  Vec b1a;
  Vec b1b;
  Vec x1a;
  Vec x1b;
  Vec ej;
  Vec J1aconj;
  Vec J1bconj;
  Vec epsSReal;
  Vec epsFReal;
  Vec epsDiff1a;
  Vec epsDiff1b;
  Vec epsDiff2;
  Vec epsMed1a;
  Vec epsMed1b;
  Vec epsMed2;
  Vec epscoef1a;
  Vec epscoef1b;
  Vec epscoef2;
  Vec ldos1agrad;
  Vec ldos1bgrad;
  Vec betagrad;
  int outputbase;
  Mat B;
  Vec vecNL;
} SFGdataGroup;


typedef struct
{
   int iz;
   int ir;
   int ic;
   int iq;
} gridInd;

// from MoperatorCyl.c or MoperatorCylwithZsym.c
PetscErrorCode MoperatorCyl(MPI_Comm comm, Mat *Aout, int m, double omega);
void expandAddress(int Nz, int Nr, int Nc, int Nq, int i, gridInd *izrcq);
int collapseAddress(int Nz, int Nr, int Nc, int Nq, gridInd *izrcq);
double complex sz(double iz, double omega);
double complex sr(double ir, double omega);
PetscErrorCode makeDh(MPI_Comm comm, Mat *Dhout, int m, double omega);
PetscErrorCode makeDe(MPI_Comm comm, Mat *Deout, int m, double omega);
PetscErrorCode makeMuinv(MPI_Comm comm, Mat *Muinvout);
PetscErrorCode GetWeightVec(Vec weight,int Nr, int Nz, int zbl);

// from SourceGeneration.c
PetscErrorCode SourceSingleSetX(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);
PetscErrorCode SourceSingleSetY(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);
PetscErrorCode SourceSingleSetZ(MPI_Comm comm, Vec J, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);
PetscErrorCode SourceSingleSetGlobal(MPI_Comm comm, Vec J, int globalpos, double amp);
PetscErrorCode SourceDuplicate(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, int scx, int scy, int scz, double amp);
PetscErrorCode SourceBlock(MPI_Comm comm, Vec *bout, int Nx, int Ny, int Nz, double hx, double hy, double hz, double lx, double ux, double ly, double uy, double lz, double uz, double amp);

// from MathTools.c
PetscErrorCode CmpVecScale(Vec vin, Vec vout, double a, double b);
PetscErrorCode CmpVecProd(Vec va, Vec vb, Vec vout);
PetscErrorCode CmpVecProdScaF(Vec v1, Vec v2, Vec v);
PetscErrorCode CmpVecDot(Vec v1, Vec v2,  double *preal, double *pimag);
PetscErrorCode ArrayToVec(double *pt, Vec V);
PetscErrorCode VecToArray(Vec V, double *pt, VecScatter scatter, IS from, IS to, Vec Vlocal, int DegFree);
PetscErrorCode ModifyMatDiag(Mat Mopr, Mat D, Vec epsF, Vec epsDiff, Vec epsMedium, Vec epspmlQ, double omega, int Nx, int Ny, int Nz);
PetscErrorCode MatSetTwoDiagonals(Mat M, Vec epsC, Mat D, double sign);
PetscErrorCode setupKSP(MPI_Comm comm, KSP *kspout, PC *pcout, int solver, int iteronly, int maxit);
PetscErrorCode SolveMatrix(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, int *its);

// from MatVecMaker.c
PetscErrorCode GetDotMat(MPI_Comm comm, Mat *Bout, int c1, int c2, int Nr, int Nz);
PetscErrorCode ImagIMat(MPI_Comm comm, Mat *Dout, int N);
PetscErrorCode CongMat(MPI_Comm comm, Mat *Cout, int N);
PetscErrorCode GetMediumVec(Vec epsmedium,int Nz, int Mz, double epsair, double epssub);
PetscErrorCode GetMediumVecwithSub(Vec epsmedium, int Nr, int Nz, int Mr, int Mz, double epsair, double epssub, int Mzslab, int mr0, int mz0);
PetscErrorCode GetRealPartVec(Vec vR, int N);
PetscErrorCode AddMuAbsorption(double *muinv, Vec muinvpml, double Qabs, int add);
PetscErrorCode GetUnitVec(Vec ej, int pol, int N);
PetscErrorCode GetRadiusVec(Vec vecRad, int Nr, int Nz, double hr, int m);
PetscErrorCode myinterp(MPI_Comm comm, Mat *Aout, int Nr, int Nz, int Mr, int Mz, int mr, int mz, int Mzslab);
PetscErrorCode expandMat(MPI_Comm comm, Mat *Aout, int DegFree, int multiplier);
PetscErrorCode myinterpmultiplier(MPI_Comm comm, Mat *Aout, int Nr, int Nz, int multiplier, int Mr, int Mz, int mr, int mz, int Mzslab);


// from Output.c
PetscErrorCode OutputVec(MPI_Comm comm, Vec x, const char *filenameComm, const char *filenameProperty);
PetscErrorCode OutputMat(MPI_Comm comm, Mat A, const char *filenameComm, const char *filenameProperty);
PetscErrorCode RetrieveVecPoints(Vec x, int Npt, int *Pos, double *ptValues);
PetscErrorCode MyCheckAndOutputInt(PetscBool flg, int CmdVar, const char *strCmdVar, const char *strCmdVarDetail);
PetscErrorCode MyCheckAndOutputDouble(PetscBool flg, double CmdVar, const char *strCmdVar, const char *strCmdVarDetail);
PetscErrorCode MyCheckAndOutputChar(PetscBool flg, char *CmdVar, const char *strCmdVar, const char *strCmdVarDetail);
PetscErrorCode GetIntParaCmdLine(int *ptCmdVar, const char *strCmdVar, const char *strCmdVarDetail);

// from mympisetup.c
int mympisetup();

// from eigsolver.c
int eigsolver(Mat M, Vec epsC, Mat D);

// form filters.c
void vecdvpow(double *u, double *v, double *dv, int n, int p);
void vectanhproj(double *u, double *v, double *dv, int n, double b, double eta);
void vecHevproj(double *u, double *v, double *dv, int n, double b, double eta);
PetscErrorCode applyfilters(int DegFree, double *epsopt, Vec epsSReal, Vec epsgrad, double pSIMP, double bproj, double etaproj);
PetscErrorCode GetH(MPI_Comm comm, Mat *Hout, int mx, int my, int mz, double s, double nR, int dim, KSP *kspHout, PC *pcHout);
PetscErrorCode RegzProj(int DegFree, double *epsopt,Vec epsSReal,Vec epsgrad,int pSIMP,double bproj,double etaproj,KSP kspH,Mat Hfilt,int *itsH);

// from PML.c
double pmlsigma(double RRT, double d);
PetscErrorCode EpsPMLFull(MPI_Comm comm, Vec epspml, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);
PetscErrorCode MuinvPMLFull(MPI_Comm comm, Vec *muinvout, int Nx, int Ny, int Nz, int Npmlx, int Npmly, int Npmlz, double sigmax, double sigmay, double sigmaz, double omega, int LowerPML);

// from ldos.c
double computeldos(KSP ksp, Mat Mopr, double omega, Vec epsFReal, Vec b, Vec Jconj, Vec x, Vec epscoef, Vec ldosgrad, int *its);

// from shg.c
double computebeta2(Vec x1, Vec x2, Vec ej, int *its, KSP ksp1, KSP ksp2, Mat Mone, Mat Mtwo, double omega1, double omega2, Vec epsFReal, Vec epscoef1, Vec epscoef2, Vec betagrad, Vec vecNL);

// from shgcrosspol.c
double computebeta2crosspol(Vec x1, Vec x2, Mat B, int *its, KSP ksp1, KSP ksp2, Mat Mone, Mat Mtwo, double omega1, double omega2, Vec epsFReal, Vec epscoef1, Vec epscoef2, Vec betagrad, Vec vecNL);

// from thg.c
double computebetathg(Vec x1, Vec x2, Vec ej, int *its, KSP ksp1, KSP ksp2, Mat Mone, Mat Mtwo, double omega1, double omega2, Vec epsFReal, Vec epscoef1, Vec epscoef2, Vec betagrad, Vec vecNL);

// from fieldfuncs.c
double funcWdotEabs(KSP ksp, Vec W, Vec epscoef, Vec grad, double omega);

// from optfuncs.c
double optldos(int DegFree, double *epsopt, double *grad, void *data);
double optfomnfc(int DegFree, double *epsopt, double *grad, void *data);

// from lvs1d.c
void struc2lsf_1d(int *struc, double *lsf, int n);
void lsf2struc_1d(double *lsf, int *struc, int n);
void edtrans1d(int *struc, double *dist, int n);
double lvs1d_opt(double (*optfunc)(int,double*,double*,void*), int DegFree, double *epsopt, double *grad, void *data, int maxeval, double dt, int steplength);
void update_struc1d(int *struc, double *v, int n, double dt, int steplength);
double find_maxabs(double *v, int n);
double min(double a, double b);
double max(double a, double b);
double test(int DegFree, double *epsopt, double *grad, void *data);

// from sfg.c
double computesfg(Vec x1a, Vec x1b, Vec x2, Vec ej, int *its, KSP ksp1a, KSP ksp1b, KSP ksp2, Mat MoneA, Mat MoneB,Mat Mtwo, double omega2, Vec epsFReal, Vec epscoef1a, Vec epscoef1b, Vec epscoef2, Vec betagrad, Vec vecNL);

// from optfuncs2.c
double optsfg(int DegFree, double *epsopt, double *grad, void *data);
