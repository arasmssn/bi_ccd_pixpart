#ifndef BI_CCD

#include "multilayer.h"

typedef struct bi_ccd_pars {
  float     N_bulk;
  float     N_f;
  float     s_f;
  float     N_b;
  float     s_b;
  float     T;
  float     t_si;
  float     overdep_bias;
  // parameters used in forming the lateral drift sigma as a function of z
  // 0 < z < t_si, equally spaced
  int       n_sigma;
  float     *z;
  float     *dp;
  double    *e;
  double    *v;
  double    *tcol;
  double    *lateral_sigma;    // expected isotropic sigma [um] due to diffusion
  // change the lateral responses to include
  // channel population (dipole) field
  // channel stop population (dirod) field
  // these two will necessarily be two dimensional lookup tables.
  // in the future these may be specified by a FITS table, particularly if 
  // computation time proves to take too long
  double    *dopevar_lat_resp; // mean lateral shift [um] response to 1% N_bulk  gradient
  double    *chanpop_lat_resp; // mean lateral shift [um] response to 1% full well content variation per micron
  double    *biasvar_lat_resp; // mean lateral shift [um] response to 1% full well content variation per micron
  // parameters used in forming the multilayer
  int       nlayer;
  optcon    *oc;
  float     *oc_const_n;
  float     *layer_thickness;
} bi_ccd_pars;

typedef float* (*gradientfunc)(float x,float y);

typedef struct ccd_runpars {
  bi_ccd_pars ccdpars;
  multilayer  ml;
  int         depleted_layer_ix;
  float       z0;
  float       reflectivity;
  int         weight_definite;
  float       poisson_weight;
  gradientfunc dopevar_grad;
  gradientfunc chanpop_grad;
  gradientfunc biasvar_grad;
} ccd_runpars;

#define N_LAYERS 1000

int    bi_ccd(ray *inc_ray,ray **detected_rays,int **n_detected,ccd_runpars *ccd_rpp );
float  abs_coeff(float lamda,float T);
float  n_silicon (float lamda);
float  mu_Si (float E,float T);
int    complain_bi_ccd (char *s);

#ifdef BI_CCD_FUNC
// these will be static variables (file scope)
int bi_ccd_print_only=0;
int bi_ccd_setup_only=0;
int bi_ccd_xray_wavelengths=0;

float e_0=8.85419e-14; // F cm^-1
float e_si=11.7;       // relative permittivity for Si
float q=1.6022e-19;    // coulombs per electron
float k=1.38066e-23;   // Joules per Kelvin

#else
// here is the list of 'hidden' variables that can alter behavior of bi_ccd()
extern int bi_ccd_print_only;
extern int bi_ccd_setup_only;
extern int bi_ccd_xray_wavelengths;

extern float e_0;
extern float e_si;
extern float q;
extern float k;
#endif

#endif
