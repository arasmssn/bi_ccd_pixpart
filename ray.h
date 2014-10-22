#ifndef _RAY_H
#define _RAY_H

#define SURF_TOLERANCE 1e-6
// for calculating gradients:
#define SURF_DELT 5.0

typedef struct vec {
  double x,y,z;
} vec;

typedef struct ray {
  vec p;
  vec k;
} ray;

typedef struct lin_trans {
  vec v_offset;
  double R_tot[3][3];
  int reverse;
  int invert;
} lin_trans;

int    complain (char *s);
void   show_ray (ray *r);
char   *show_vector (char *s,vec *v);
int    ray_surface_collapse (ray *r,double (*func)(vec *v));
int    surface_normal (vec *v,double (*func)(vec *v),vec *n);
void   figure_error (vec *v,float err1,float err2);
double dot_prod (vec *v1,vec *v2);
void   cross_prod (vec *v1,vec *v2,vec *vprod);
int    reflect_ray(vec *k,vec *surf_norm);
void   vec_diff (vec *v1,vec *v2,vec *vecdif);
void   vec_add (vec *v1,vec *v2,vec *vecsum);
void   unitvec (vec *v);
void   scalevec (vec *v,float s);
void   cpvec (vec *vfrom,vec *vto);
double modulus (vec *v);
int    refract_ray(vec *k,vec *normal,float n_from,float n_to);
void   fill_transform_specs (char *ts,lin_trans *tf);
void   tran_ray(ray *aray,lin_trans *tf,int direction);
void   parse_transform_subargs(char *transform_string,int *argc,char **argv[]);
// double zernike (float rho,float theta,int n,int m);

#endif
