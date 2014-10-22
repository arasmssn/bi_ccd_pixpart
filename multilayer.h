#ifndef MULTILAYER

typedef struct {
  double comp[2];
} complex;

typedef void (*optcon)(float,complex*);

void    set_ml_Si_Temp(float T);
char    *show_complex (char *s,complex *c);
float    cpx_modulus  (complex *num);
complex *cpx_set      (complex *c,float re,float im);
complex *cpx_sin      (complex *sin_theta,complex *theta);
complex *cpx_negate   (complex *num);
complex *cpx_conjugate(complex *num);
complex *cpx_product  (complex *a,complex *b);
complex *cpx_ratio    (complex *a,complex *b);
complex *cpx_inverse  (complex *a);
complex *cpx_sqrt     (complex *a);
complex *cpx_increment(complex *a,complex *b);

void    cpx_2x2_mat_prod (complex *a[][2], complex *b[][2], complex *prod[][2]);
void    cpx_2x2_mat_inv  (complex *a[][2], complex *inv[][2]);
complex *cpx_2x2_mat_det  (complex *a[][2], complex *det);

void    display_mat(complex *a[][2]);

void glass (float lamda,complex *n);
void vacuum (float lamda,complex *n);
void air (float lamda,complex *n);
void HfO2(float lamda,complex *n);
void Si  (float lamda,complex *n);
void MgF2(float lambda,complex *n);
void H2O(float lambda,complex *n);
void silicone_oil  (float lambda,complex *n);
void SiO2 (float lamda,complex *n);
void Ta2O5(float lambda,complex *n);
void TiO2 (float lambda,complex *n);
void const_n_func(float lambda,complex *n);
void metal(float lambda,complex *n);
void al_metal(float lambda,complex *n);
void au_metal(float lambda,complex *n);
void ag_metal(float lambda,complex *n);

void Au_cpxn(float lambda,complex *n);

void Si3N4_cpxn(float lambda,complex *n);
void Si_poly_cpxn(float lambda,complex *n);

optcon get_material(char *s);
void   ml_complain(char *s);

#ifdef MULTILAYER_HOME

char *usage_str;

int n_ml_media=19;

optcon ml_media[]={vacuum,
		   air,
		   glass,
		   HfO2,
		   Si,
		   MgF2,
		   silicone_oil,
		   SiO2,
		   Ta2O5,
		   TiO2,
		   H2O,
		   metal,
		   al_metal,
		   ag_metal,
		   au_metal,
		   Au_cpxn,
		   Si3N4_cpxn,
		   Si_poly_cpxn,
		   const_n_func};

char *ml_media_names[]={"vacuum",
			"air",
			"glass",
			"HfO2",
			"Si",
			"MgF2",
			"silicone_oil",
			"SiO2",
			"Ta2O5",
			"TiO2",
			"H2O",
			"metal",
			"Al",
			"Ag",
			"Au_old",
			"Au",
			"Si3N4",
			"SiPOLY",
			"const"};
#else

extern char    *usage_str;
extern int     n_ml_media;
extern optcon  ml_media[];
extern char   *ml_media_names[];

#endif

typedef struct {
  optcon optical_constants;
  float theta,lambda;
  float const_n_val;
  complex cpx_sin_theta;
  complex n;
} media_pars;

typedef struct {
  media_pars *media_p[2];
  float rho[2];
  float tau[2];
  complex cpx_rho[2];
  complex cpx_tau[2];
  float a,b;
  complex cpx_a,cpx_b;
  complex H_mx[2][2][2];  // H_mx is complex, one for each polarization
  float re_H_mx[2][2][2]; // re_H_mx is real but one for each polarization
} interface_pars;

typedef struct {
  interface_pars *iface_p[2];
  float thickness;
  float Dq[2];        // flux stopped in layer for each polarization
  float Dq_ave;       // flux stopped in layer for unpolarized light
  complex beta;
  complex L_mx[2][2]; // L_mx is complex and identical for each polarization
  complex S1Q[2][2][2]; // partial/running stack matrix for computing detection
  complex ErQ_div_Er1[2]; // normalized right going amplitude (@ left end)
  complex ElQ_div_Er1[2]; // normalized left going amplitude (@ left end)
} layer_pars;

typedef struct {
  // polarization index convention: 
  // 0 => sigma (E perpendicular to plane of incidence) 
  // 1 => pi    (E in plane of plane of incidence)
  float          lambda;
  float          theta;
  layer_pars     *lp;
  media_pars     *mp;
  interface_pars *ip;
  int            nlayer;
  complex S[2][2][2]; // the stack matrix
  complex tau[2];
  complex rho[2];
  float R[2];
  float T[2];
  float R_ave;
  float T_ave;
  float beta_total;
  int   eff_nlayer;
} multilayer;

void   compute_layer_internal_reflectivities(multilayer *mlstruct,int layer_ix,
					     complex rho_q[],complex rho_qp[]);
void   compute_multilayer(float theta,float lambda,multilayer *mlstruct);
void   init_multilayer(multilayer *ml,
		     int nlayer,float *layer_thickness,optcon *oc,
		     float *oc_const_n);

#define MULTILAYER
#endif
