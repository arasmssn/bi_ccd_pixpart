#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "numrec/include/nr.h"
#include "ray.h"
#undef BI_CCD_FUNC
#include "bi_ccd.h"
#include "field_superpose.h"

typedef struct {
  double *sigma_resp;
  double *tcol;
  double *emag;
  double *potential;
  vec    *pos;
} driftstruct;

enum direction {FORWARD=0,BACKWARD=1};
enum boundarytype {BND_EITHER=0,BND_COLUMN=1,BND_ROW=2};

void target_pix(int x,int y,vec *ps,bi_ccd_pars *pars,
		multipole_stack *msi,double *e,int n_vals,
		enum boundarytype bndtype);
void drift2pix (vec *ps,bi_ccd_pars *pars,multipole_stack *msi,
		double *e,int n_vals,int *xpix,int *ypix);
int multipole_stack_pinch(vec *p1,vec *p2,vec *pinch,
			  enum boundarytype bndtype,
			  bi_ccd_pars *pars,double *e1,
			  multipole_stack *msi);
void drift (vec *ps,driftstruct *ds,
	    bi_ccd_pars *pars,multipole_stack *msi,double *e,
	    int n_vals,enum direction reverse,int *iter);
void dipole_3d (vec *x_vec,vec *p_vec,vec *x0_vec,vec *dummy,vec *field);
void dipole_2d (vec *x_vec,vec *p_vec,vec *x0_vec,vec *sym_axis,vec *field);
void trunc_dipole_2d (vec *x_vec,vec *p_vec,vec *x0_vec,
		      vec *sym_axis,vec *field);
void dipolegrid_2d_orig (vec *x_vec,vec *p_vec,vec *x0_vec,
			 vec *sym_axis,vec *field);
void dipolegrid_2d (vec *x_vec,vec *p_vec,vec *x0_vec,
		    vec *sym_axis,vec *field);

typedef struct {
  float dipole_moment; // product of the number of charges with 2x the position
                       // relative to the nearest equipotential surface
  float startpix;
  int   npix; // total number of pixels to be bounded uniformly. 
              // if npix=0 this is a single row or column at 
              // the lower boundary of that coordinate (units are pixels)
  float theta; // angle of moment vector wrt normal, prior to any rotation by phi
  float phi; // angle corresponding to symmetry axis in degrees: 0=x, 90=y
} dipole_array_struct;

typedef struct {
  float dipole_moment; // product of the number of charges with 2x the position
                       // relative to the nearest equipotential surface
  float startpix_x,startpix_y;
  float theta; // angle of moment vector wrt normal, prior to any rotation by phi
  float phi; // angle corresponding to symmetry axis in degrees: 0=x, 90=y
} truncated_dipole_struct;

typedef struct {
  dipole_array_struct *dpgs;
  int ndpg;
  int n_dpg_chunk;
} dpg_mgmt;

typedef struct {
  truncated_dipole_struct *tdps;
  int ntdp;
  int n_tdp_chunk;
} tdp_mgmt;

typedef struct {
  dipole_array_struct *dpas;
  int ndpa;
  int n_dpa_chunk;
} dpa_mgmt;

typedef struct {
  float dipole_moment;
  vec   p;
} dipole_list_struct;

typedef struct {
  dipole_list_struct *dls;
  int ndls;
  int n_dls_chunk;
} dl_mgmt;

typedef struct {
  vec slice[2];
  int n_elem;
} slice_subset;

typedef struct {
  slice_subset *slcs;
  int nss;
  int n_ss_chunk;
} slice_mgmt;

typedef struct {
  float limits[2]; // coordinates (in pixels) between which to find the boundary locus 
  int pix; // boundary will be found between pix-1 & pix
  enum boundarytype way;
  int n_elem;
} bnd_subset; // pixel boundary management

typedef struct {
  bnd_subset *bnds;
  int nbnds;
  int n_bnd_chunk;
} bnd_mgmt;

typedef struct {
  int pixcoord[2]; // coordinate pairs (in pixels) of pixel for which to find the boundary locus. corner will be found between pixcoord-1 & pixcoord
  enum boundarytype way;
  double *dist_along_edge;
  double *edge_distortion;
  vec corner[2];
  int n_elem;
} crn_subset; // pixel boundary management

typedef struct {
  crn_subset *crns;
  int ncrns;
  int n_crn_chunk;
} crn_mgmt;

typedef struct {
  float transv_pixcoord; // boundary will be found between pix & pix+1
  int pix[2]; // coordinates (in pixels) between which to find the boundary locus 
  enum boundarytype way;
} wid_subset; // pixel boundary management

typedef struct {
  wid_subset *wids;
  int nwids;
  int n_wid_chunk;
} wid_mgmt;

void locate_pix_edge (crn_subset *crn,ccd_runpars *ccd_rpp,multipole_stack *msi);

char *usage_str=
  "usage::\n"
  "bi_ccd_pixpart [plane parallel field pars]\n"
  "[plane parallel field pars] can be expanded into the following list\n"
  "of input parameters, which describe the plane parallel Poisson solution:\n"
  "     -sf <frontside doping scale> -sb <backside doping scale>\n"
  "     -Nf <frontside doping density> -Nb <backside doping density>\n"
  "     -N  <bulk doping density> -t <device thickness> -B <backside bias>\n"
  "     -g <infinite_2d_dipole_array_moment0> <phi0>\n"
  "     -g <infinite_2d_dipole_array_moment1> <phi1>\n"
  "     -G <infinite_2d_dipole_array_moment0> <subpix_phase0> <theta0> <phi0>\n"
  "     -G <infinite_2d_dipole_array_moment1> <subpix_phase1> <theta1> <phi1>\n"
  "     -d <2d_dipole_array_moment0> <array_start0> <npix0> <phi0>\n"
  "     -d <2d_dipole_array_moment1> <array_start1> <npix1> <phi1> ..\n"
  "     -D <2d_dipole_array_moment0> <array_startphase0> <npix0> <theta0> <phi0> ..\n"
  "     -D <2d_dipole_array_moment1> <array_startphase1> <npix1> <theta1> <phi1> ..\n"
  "     -r <trunc_2d_dipole_moment0> <truncpix_x0> <truncpix_y0> <phi0>\n"
  "     -r <trunc_2d_dipole_moment1> <truncpix_x1> <truncpix_y1> <phi1>\n"
  "     -c <3d_dipole_moment0> <dipole0_pix_x> <dipole0_pix_y>\n"
  "     -c <3d_dipole_moment1> <dipole1_pix_x> <dipole1_pix_y>\n"
  "     -S <slice0_startx> <slice_starty> <slice_stopx> <slice_stopy> <npts>\n"
  "     -S <slice1_startx> <slice_starty> <slice_stopx> <slice_stopy> <npts> ..\n"
  "     -b <bnd0_pixel> <bnd0_lo_lim> <bnd0_hi_lim> <bnd0_n_sample> <dir[x|y]>\n"
  "     -b <bnd1_pixel> <bnd1_lo_lim> <bnd1_hi_lim> <bnd1_n_sample> <dir[x|y]>..\n"
  "     -k <crn0_pixel_x0> <crn0_pixel_y0> <crn0_n_sample> <dir[x|y]>\n"
  "     -k <crn1_pixel_x0> <crn1_pixel_y0> <crn1_n_sample> <dir[x|y]>..\n"
  "     -w <width0_transv_pixelcoord> <w0_lo_lim> <w0_hi_lim> <dir[x|y]>\n"
  "     -w <width1_transv_pixelcoord> <w1_lo_lim> <w1_hi_lim> <dir[x|y]>..\n"
  "     -n <image pair count>\n";



int main(int argc,char *argv[]) {
  
  int nim=6;

  wid_mgmt wid;
  wid.wids=NULL;
  wid.nwids=0;
  wid.n_wid_chunk=10;
  wid.wids=(wid_subset*)
    malloc(wid.n_wid_chunk*sizeof(wid_subset));

  dpg_mgmt dpgm;

  dpgm.dpgs=NULL;
  dpgm.ndpg=0;
  dpgm.n_dpg_chunk=10;
  dpgm.dpgs=(dipole_array_struct*)
    malloc(dpgm.n_dpg_chunk*sizeof(dipole_array_struct));

  tdp_mgmt tdpm;

  tdpm.tdps=NULL;
  tdpm.ntdp=0;
  tdpm.n_tdp_chunk=10;
  tdpm.tdps=(truncated_dipole_struct*)
    malloc(tdpm.n_tdp_chunk*sizeof(truncated_dipole_struct));

  dpa_mgmt dpam;

  dpam.dpas=NULL;
  dpam.ndpa=0;
  dpam.n_dpa_chunk=10;
  dpam.dpas=(dipole_array_struct*)
    malloc(dpam.n_dpa_chunk*sizeof(dipole_array_struct));
  
  dl_mgmt dlm;

  dlm.dls=NULL;
  dlm.ndls=0;
  dlm.n_dls_chunk=10;
  dlm.dls=(dipole_list_struct*)
    malloc(dlm.n_dls_chunk*sizeof(dipole_list_struct));

  slice_mgmt slm;
  slm.nss=0;
  slm.n_ss_chunk=10;
  slm.slcs=(slice_subset*)
    malloc(slm.n_ss_chunk*sizeof(slice_subset));

  bnd_mgmt bnd;
  bnd.nbnds=0;
  bnd.n_bnd_chunk=10;
  bnd.bnds=(bnd_subset*)
    malloc(bnd.n_bnd_chunk*sizeof(bnd_subset));

  crn_mgmt crn;
  crn.ncrns=0;
  crn.n_crn_chunk=10;
  crn.crns=(crn_subset*)
    malloc(crn.n_crn_chunk*sizeof(crn_subset));

  ccd_runpars ccd_rp;

  ccd_rp.ccdpars.N_bulk       = 1e12;
  ccd_rp.ccdpars.N_b          = 0;
  ccd_rp.ccdpars.s_b          = 0.2e-4;
  ccd_rp.ccdpars.N_f          = -5e16; // this frontside doping parameter choice
  ccd_rp.ccdpars.s_f          = 1e-5;  // gives a buried channel 350nm deep
  ccd_rp.ccdpars.t_si         = 100e-4;
  ccd_rp.ccdpars.overdep_bias = 0;
  ccd_rp.ccdpars.T            = 173;

  ccd_rp.reflectivity   = 0;
  ccd_rp.z0             = 0;
  ccd_rp.poisson_weight = 0;
  ccd_rp.weight_definite= 1;
  ccd_rp.chanpop_grad = NULL; 
  ccd_rp.dopevar_grad = NULL; 
  ccd_rp.biasvar_grad = NULL; 

  // e- * cm moment (100% fw) - suppose buried channel is 10nm deep
  double p_moment=100000*q*(1.0)*1000e-7; 
  // e- * cm / cm moment (analog of 1% fw)
  double xi_moment=100000*q*1e3*(1.0*1e-3)/10.0; 
  
  vec nullvec={0.0,0.0,0.0};

  while (--argc && argv++) {
    switch (argv[0][0]) {
    case '-':
      switch (argv[0][1]) {
      case 's':      case 'N':      case 't':      case 'T':      case 'z':
      case 'R':      case 'B':      case 'p':      case 'n':
	if (argc<2) goto bad_args;
	break;
      case 'S':
      case 'b':
      case 'D':
	if (argc<6) goto bad_args; // 6 corresponds to 5 arguments following
	break;
      case 'r':
      case 'd':
      case 'w':      
      case 'k':
      case 'G':
	if (argc<5) goto bad_args; // 5 corresponds to 4 arguments following
	break;
      case 'c':
	if (argc<4) goto bad_args; // 4 corresponds to 3 arguments following
	break;
      case 'g':
	if (argc<3) goto bad_args; // 3 corresponds to 2 arguments following
	break;
      case 'P':
	bi_ccd_print_only=1;
	break;
      case 'h': // no following arguments necessary..
      case 'e':
	break;
      default:
	goto bad_switch;
	break;
      }
      switch(argv[0][1]) {
      case 'n':
	argc--;argv++;	nim = atoi(argv[0]);
	break;
      case 'S':
	// fill in the z location later
	if (slm.nss >= slm.n_ss_chunk) {
	  slm.n_ss_chunk *= 2;
	  slm.slcs=(slice_subset*)
	    realloc(slm.slcs,slm.n_ss_chunk*sizeof(slice_subset));
	}
	cpvec(&nullvec,&slm.slcs[slm.nss].slice[0]);
	cpvec(&nullvec,&slm.slcs[slm.nss].slice[1]);
	argc--;argv++;	slm.slcs[slm.nss].slice[0].x = atof(argv[0]);
	argc--;argv++;	slm.slcs[slm.nss].slice[0].y = atof(argv[0]);
	argc--;argv++;	slm.slcs[slm.nss].slice[1].x = atof(argv[0]);
	argc--;argv++;	slm.slcs[slm.nss].slice[1].y = atof(argv[0]);
	argc--;argv++;	slm.slcs[slm.nss].n_elem = atoi(argv[0]);
	slm.nss++;
	break;
      case 'w':
	if (wid.nwids >= wid.n_wid_chunk) {
	  wid.n_wid_chunk *= 2;
	  wid.wids=(wid_subset*)
	    realloc(wid.wids,wid.n_wid_chunk*sizeof(wid_subset));
	}
	argc--;argv++;	wid.wids[wid.nwids].transv_pixcoord = atof(argv[0]);
	argc--;argv++;	wid.wids[wid.nwids].pix[0] = atoi(argv[0]);
	argc--;argv++;	wid.wids[wid.nwids].pix[1] = atoi(argv[0]);
	argc--;argv++;	
	switch (argv[0][0]) {
	case 'x':	case 'X':	  
	  wid.wids[wid.nwids].way=BND_COLUMN;	  break;
	case 'y':	case 'Y':	  
	  wid.wids[wid.nwids].way=BND_ROW;	  break;
	default:
	  fprintf(stderr,"width scan direction for boundary search needs to be "
		  "specified as either x or y\nx specifies determining "
		  "boundary between adjacent rows\ny specifies determining "
		  "boundary between adjacent columns.\nexiting..\n");
	  exit(1);
	  break;
	}
	wid.nwids++;
	break;
      case 'b':
	// fill in the bnd_mgmt structure for later use
	if (bnd.nbnds >= bnd.n_bnd_chunk) {
	  bnd.n_bnd_chunk *= 2;
	  bnd.bnds=(bnd_subset*)
	    realloc(bnd.bnds,bnd.n_bnd_chunk*sizeof(bnd_subset));
	}
	argc--;argv++;	bnd.bnds[bnd.nbnds].pix = atoi(argv[0])-1;
	argc--;argv++;	bnd.bnds[bnd.nbnds].limits[0] = atof(argv[0]);
	argc--;argv++;	bnd.bnds[bnd.nbnds].limits[1] = atof(argv[0]);
	argc--;argv++;	bnd.bnds[bnd.nbnds].n_elem    = atoi(argv[0]);
	argc--;argv++;	
	switch (argv[0][0]) {
	case 'x':	case 'X':	  
	  bnd.bnds[bnd.nbnds].way=BND_ROW;	  break;
	case 'y':	case 'Y':	  
	  bnd.bnds[bnd.nbnds].way=BND_COLUMN;	  break;
	default:
	  fprintf(stderr,"scan direction for boundary search needs to be "
		  "specified as either x or y\nx specifies determining "
		  "boundary between adjacent rows\ny specifies determining "
		  "boundary between adjacent columns.\nexiting..\n");
	  exit(1);
	  break;
	}
	bnd.nbnds++;
	break;
      case 'k':
	// fill in the bnd_mgmt structure for later use
	if (crn.ncrns >= crn.n_crn_chunk) {
	  crn.n_crn_chunk *= 2;
	  crn.crns=(crn_subset*)
	    realloc(crn.crns,crn.n_crn_chunk*sizeof(crn_subset));
	}
	argc--;argv++;	crn.crns[crn.ncrns].pixcoord[0] = atoi(argv[0]);
	argc--;argv++;	crn.crns[crn.ncrns].pixcoord[1] = atoi(argv[0]);
	argc--;argv++;	crn.crns[crn.ncrns].n_elem      = atoi(argv[0]);
	crn.crns[crn.ncrns].dist_along_edge=
	  (double*)malloc(crn.crns[crn.ncrns].n_elem*sizeof(double));
	crn.crns[crn.ncrns].edge_distortion=
	  (double*)malloc(crn.crns[crn.ncrns].n_elem*sizeof(double));
	argc--;argv++;	
	switch (argv[0][0]) {
	case 'x':	case 'X':	  
	  crn.crns[crn.ncrns].way=BND_ROW;	  break;
	case 'y':	case 'Y':	  
	  crn.crns[crn.ncrns].way=BND_COLUMN;	  break;
	default:
	  fprintf(stderr,"scan direction specification for corner-to-corner "
		  "boundary search needs to be specified as either x or y.\n"
		  "x specifies determining boundary between adjacent "
		  "rows (and a trace between [x,y] and [x+1,y])\n"
		  "y specifies determining boundary between adjacent columns (and a trace between [x,y] and [x,y+1]).\n"
		  "exiting..\n");
	  exit(1);
	  break;
	}
	crn.ncrns++;
	break;
      case 'd':
	if (dpam.ndpa >= dpam.n_dpa_chunk) {
	  dpam.n_dpa_chunk *= 2;
	  dpam.dpas=(dipole_array_struct*)
	    realloc(dpam.dpas,dpam.n_dpa_chunk*sizeof(dipole_array_struct));
	}
	// populate the next dipole_array_struct and increment counter
	argc--;argv++;	dpam.dpas[dpam.ndpa].dipole_moment=atof(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].startpix=atof(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].npix=atoi(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].phi=atof(argv[0]);
	dpam.dpas[dpam.ndpa].theta=0; // normal to surface
	dpam.ndpa++;
	break;
      case 'D':
	if (dpam.ndpa >= dpam.n_dpa_chunk) {
	  dpam.n_dpa_chunk *= 2;
	  dpam.dpas=(dipole_array_struct*)
	    realloc(dpam.dpas,dpam.n_dpa_chunk*sizeof(dipole_array_struct));
	}
	// populate the next dipole_array_struct and increment counter
	argc--;argv++;	dpam.dpas[dpam.ndpa].dipole_moment=atof(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].startpix=atof(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].npix=atoi(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].theta=atof(argv[0]);
	argc--;argv++;	dpam.dpas[dpam.ndpa].phi=atof(argv[0]);
	dpam.ndpa++;
	break;
      case 'g':
	if (dpgm.ndpg >= dpgm.n_dpg_chunk) {
	  dpgm.n_dpg_chunk *= 2;
	  dpgm.dpgs=(dipole_array_struct*)
	    realloc(dpgm.dpgs,dpgm.n_dpg_chunk*sizeof(dipole_array_struct));
	}
	// populate the next dipole_array_struct and increment counter
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].dipole_moment=atof(argv[0]);
	// dipole_array_struct is also used to designate an infinite 
	// 2D dipole grid, except that parameters "startpix" and "npix"
	// are not used yet. Perhaps when this works toward supporting
	// truncated grids..
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].phi=atof(argv[0]);
	dpgm.dpgs[dpgm.ndpg].startpix=0; // aligned with nominal grid
	dpgm.dpgs[dpgm.ndpg].theta=0; // normal to surface
	dpgm.ndpg++;
	break;
      case 'G':
	if (dpgm.ndpg >= dpgm.n_dpg_chunk) {
	  dpgm.n_dpg_chunk *= 2;
	  dpgm.dpgs=(dipole_array_struct*)
	    realloc(dpgm.dpgs,dpgm.n_dpg_chunk*sizeof(dipole_array_struct));
	}
	// populate the next dipole_array_struct and increment counter
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].dipole_moment=atof(argv[0]);
	// dipole_array_struct is also used to designate an infinite 
	// 2D dipole grid, except that parameters "startpix" and "npix"
	// are not used yet. Perhaps when this works toward supporting
	// truncated grids..
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].startpix=atof(argv[0]);
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].theta=atof(argv[0]);
	argc--;argv++;	dpgm.dpgs[dpgm.ndpg].phi=atof(argv[0]);
	dpgm.ndpg++;
	break;
      case 'r':
	if (tdpm.ntdp >= tdpm.n_tdp_chunk) {
	  tdpm.n_tdp_chunk *= 2;
	  tdpm.tdps=(truncated_dipole_struct*)
	    realloc(tdpm.tdps,tdpm.n_tdp_chunk*sizeof(truncated_dipole_struct));
	}
	// populate the next dipole_array_struct and increment counter
	argc--;argv++;	tdpm.tdps[tdpm.ntdp].dipole_moment=atof(argv[0]);
	argc--;argv++;	tdpm.tdps[tdpm.ntdp].startpix_x=atof(argv[0]);
	argc--;argv++;	tdpm.tdps[tdpm.ntdp].startpix_y=atof(argv[0]);
	argc--;argv++;	tdpm.tdps[tdpm.ntdp].phi=atof(argv[0]);
	tdpm.tdps[tdpm.ntdp].theta=0; // normal to surface
	tdpm.ntdp++;
	break;
      case 'c':
	if (dlm.ndls >= dlm.n_dls_chunk) {
	  dlm.n_dls_chunk *= 2;
	  dlm.dls=(dipole_list_struct*)
	    realloc(dlm.dls,dlm.n_dls_chunk*sizeof(dipole_list_struct));
	}
	// populate the next dipole_array_struct and increment counter
	cpvec(&nullvec,&dlm.dls[dlm.ndls].p);
	argc--;argv++;	dlm.dls[dlm.ndls].dipole_moment=atof(argv[0]);
	argc--;argv++;	dlm.dls[dlm.ndls].p.x=(float)atoi(argv[0]);
	argc--;argv++;	dlm.dls[dlm.ndls].p.y=(float)atoi(argv[0]);
	dlm.ndls++;
	break;
      case 'p':
	argc--;argv++;	ccd_rp.poisson_weight=atof(argv[0]);
	break;
      case 'B':
	argc--;argv++;	ccd_rp.ccdpars.overdep_bias=atof(argv[0]);
	break;
      case 'R':
	argc--;argv++;	ccd_rp.reflectivity=atof(argv[0]);
	break;
      case 'z':
	argc--;argv++;	ccd_rp.z0=atof(argv[0]);
	break;
      case 'T':
	argc--;argv++;	ccd_rp.ccdpars.T=atof(argv[0]);
	break;
	// thicknesses specified in microns, stored in cm
      case 't':
	argc--;argv++;	ccd_rp.ccdpars.t_si=atof(argv[0])/1e4; 
	break;
      case 's': 
	switch(argv[0][2]) {
	case 'b':
	  argc--;argv++;	  ccd_rp.ccdpars.s_b=atof(argv[0])/1e4;
	  break;
	case 'f':
	  argc--;argv++;	  ccd_rp.ccdpars.s_f=atof(argv[0])/1e4;
	  break;
	default:
	  goto bad_switch;
	  break;
	}
	break;
      case 'N':
	switch(argv[0][2]) {
	case 'b':
	  argc--;argv++;	  ccd_rp.ccdpars.N_b=atof(argv[0]);
	  break;
	case 'f':
	  argc--;argv++;	  ccd_rp.ccdpars.N_f=atof(argv[0]);
	  break;
	case 0:
	  argc--;argv++;	  ccd_rp.ccdpars.N_bulk=atof(argv[0]);
	  break;
	default:
	  goto bad_switch;
	  break;
	}
	break;
      }
      break;
    default:
      break;
    }
  }
  // if control reaches here, we're ready to proceed. skip over the error cases
  goto resume;

 bad_switch:
  fprintf(stderr,"unknown switch: %s",argv[0]);
  complain_bi_ccd(NULL);
  exit(1);
 bad_args:
  fprintf(stderr,"argument(s) expected after switch: %s",argv[0]);
  complain_bi_ccd(NULL);
  exit(1);

 resume:
  fprintf(stderr,"%s",usage_str);
  {
    // prior to calling bi_ccd need to initialize a multilayer
    {
      int    *nlayer=&ccd_rp.ccdpars.nlayer;
      optcon **oc=&ccd_rp.ccdpars.oc;
      float  **oc_const_n=&ccd_rp.ccdpars.oc_const_n;
      float  **layer_thickness=&ccd_rp.ccdpars.layer_thickness;
      multilayer *ml=&ccd_rp.ml;
      *nlayer=1;
      *oc=(optcon*)malloc((*nlayer+2)*sizeof(optcon));
      *oc_const_n=(float*)malloc((*nlayer+2)*sizeof(float));
      *layer_thickness=(float*)malloc((*nlayer)*sizeof(float));
        
      int    ix;
        
      ml->mp=NULL;    ml->ip=NULL;    ml->lp=NULL;
        
      ix=0; (*oc)[ix]=get_material("vacuum");
      (*layer_thickness)[ix]= ccd_rp.ccdpars.t_si*1e7;   
      //
      ccd_rp.depleted_layer_ix=ix;
      ix++; (*oc)[ix]=get_material("Si");
      ix++; (*oc)[ix]=get_material("vacuum");

      // initialize
      init_multilayer(ml,*nlayer,*layer_thickness,*oc,*oc_const_n);
    }
    ray aray,*output_rays;
    int *n_rays;
    bi_ccd_xray_wavelengths=0;
    bi_ccd_print_only=0;
    bi_ccd_setup_only=1;
    bi_ccd(&aray,&output_rays,&n_rays,&ccd_rp);
  }
  // now the computed field profile resides within ccd_rp


  // form contributions to the clock- and channel stop arrays of 2D dipoles
  // include any 3D dipole(s) specified
  // do this layer by layer, where each layer represents the next image
  // in the image expansion

  multipole_stack *msi=NULL;

  // establish a multipole background field to reduce roundoff and/or asymmetry
  // in the arrangement of dipoles and their images

  multipole_stack_init(&msi);

  {
    int dpa_ix,dpg_ix,dls_ix,tdp_ix;
    int im;
    
    for (im=0;im<=nim;im++) {
      // dipole grids
      for (dpg_ix=0;dpg_ix<dpgm.ndpg;dpg_ix++) {
	dipole_array_struct *dps=&dpgm.dpgs[dpg_ix];
	float deg=atan2(1,1)/45.0;
	float cs=cos(dps->phi*deg);
	float sn=sin(dps->phi*deg);
	if (dps->phi == 90) {
	  cs=0;
	  sn=1;
	}
	// establish axis of symmetry
	vec sym_axis={0,1,0};
	vec step={0.01/10.0,0,0};
	vec startpos;
	cpvec(&step,&startpos);
	scalevec(&startpos,(float)dps->startpix);

	{
	  float xc;
	  float yc;
	  vec *vp;

	  vp=&sym_axis;
	  xc=vp->x;             yc=vp->y;
	  vp->x=+xc*cs-yc*sn;	vp->y=+xc*sn+yc*cs;

	  // this should be unnecessary
	  vp=&startpos;
	  xc=vp->x;         	yc=vp->y;
	  vp->x=+xc*cs-yc*sn;	vp->y=+xc*sn+yc*cs;
	}
	vec pos;
	vec z_unit={0,0,1};
	vec dipole_unit={cs*sin(dps->theta*deg),
			 sn*sin(dps->theta*deg),
			 cos(dps->theta*deg)};
	vec z_image;
	int each;

	for (each=0;each<2;each++) { // each=0 is for the image on the other
	  // side of the channel. each=1 is for the image on the other side
	  // of the backside electrode
	  if ((im==0) && (each==1)) continue;   // avoid counting twice
	  if ((im==nim) && (each==0)) continue; // special attention to field at the backside.
	  cpvec(&z_unit,&z_image);
	  scalevec(&z_image,(float)2*im*(2*each-1)*ccd_rp.ccdpars.t_si);
	  cpvec(&startpos,&pos);
	  pos.z = z_image.z;
	  vec dipole2d;
	  //	  cpvec(&z_unit,&dipole2d);
	  cpvec(&dipole_unit,&dipole2d);
	  scalevec(&dipole2d,dps->dipole_moment*xi_moment);
	  multipole_stack_append(&msi,dipolegrid_2d,&dipole2d,&pos,&sym_axis);
	}
      }

      // truncated dipoles

      for (tdp_ix=0;tdp_ix<tdpm.ntdp;tdp_ix++) {
	truncated_dipole_struct *tdp=&tdpm.tdps[tdp_ix];

	float deg=atan2(1,1)/45.0;
	float cs=cos(tdp->phi*deg);
	float sn=sin(tdp->phi*deg);
	if (tdp->phi == 90) {
	  cs=0;
	  sn=1;
	}
	// establish axis of symmetry to be rotated by phi later
	vec trunc_axis={0,1,0};
	vec tdp_pos={tdp->startpix_x,tdp->startpix_y,0}; // units are pixels
	scalevec(&tdp_pos,0.01/10.0); // position of the truncation, now in cm

	{
	  float xc;
	  float yc;
	  vec  *vp;

	  vp=&trunc_axis;
	  xc=vp->x;             yc=vp->y;
	  // the following choice is so that phi=0 corresponds to channel stops (y) and phi=90 corresponds to x.
	  vp->x=+xc*cs+yc*sn;	vp->y=-xc*sn+yc*cs;
	}

	vec pos;
	vec z_unit={0,0,1};
	// the following choice is consistent with definition above.
	vec dipole_unit={+cs*sin(tdp->theta*deg),
			 -sn*sin(tdp->theta*deg),
			 +cos(tdp->theta*deg)};
	vec z_image;
	int each;

	for (each=0;each<2;each++) { // each=0 is for the image on the other
	  // side of the channel. each=1 is for the image on the other side
	  // of the backside electrode
	  if ((im==0) && (each==1)) continue;   // avoid counting twice
	  if ((im==nim) && (each==0)) continue; // special attention to field at the backside.
	  cpvec(&z_unit,&z_image);
	  scalevec(&z_image,(float)2*im*(2*each-1)*ccd_rp.ccdpars.t_si);
	  cpvec(&tdp_pos,&pos);
	  pos.z = z_image.z;
	  vec dipole2d;
	  // cpvec(&z_unit,&dipole2d);
	  cpvec(&dipole_unit,&dipole2d);
	  scalevec(&dipole2d,tdp->dipole_moment*xi_moment);
	  multipole_stack_append(&msi,trunc_dipole_2d,&dipole2d,&pos,&trunc_axis);
	}
      }

      // dipole arrays

      for (dpa_ix=0;dpa_ix<dpam.ndpa;dpa_ix++) {
	dipole_array_struct *dpa=&dpam.dpas[dpa_ix];
	int ix;
	float deg=atan2(1,1)/45.0;
	float cs=cos(dpa->phi*deg);
	float sn=sin(dpa->phi*deg);
	if (dpa->phi == 90) {
	  cs=0;
	  sn=1;
	}
	// establish axis of symmetry
	vec sym_axis={0,1,0};
	vec step={0.01/10.0,0,0};
	vec startpos;
	cpvec(&step,&startpos);
	scalevec(&startpos,(float)dpa->startpix);

	{
	  float xc;
	  float yc;
	  vec *vp;

	  vp=&sym_axis;
	  xc=vp->x;             yc=vp->y;
	  vp->x=+xc*cs-yc*sn;	vp->y=+xc*sn+yc*cs;

	  vp=&step;
	  xc=vp->x;         	yc=vp->y;
	  vp->x=+xc*cs-yc*sn;	vp->y=+xc*sn+yc*cs;

	  vp=&startpos;
	  xc=vp->x;         	yc=vp->y;
	  vp->x=+xc*cs-yc*sn;	vp->y=+xc*sn+yc*cs;
	}
	vec pos;
	vec z_unit={0,0,1};
	vec dipole_unit={cs*sin(dpa->theta*deg),
			 sn*sin(dpa->theta*deg),
			 cos(dpa->theta*deg)};
	vec z_image;
	int each;

	for (each=0;each<2;each++) { // each=0 is for the image on the other
	  // side of the channel. each=1 is for the image on the other side
	  // of the backside electrode
	  if ((im==0) && (each==1)) continue;   // avoid counting twice
	  if ((im==nim) && (each==0)) continue; // special attention to field at the backside.
	  cpvec(&z_unit,&z_image);
	  scalevec(&z_image,(float)2*im*(2*each-1)*ccd_rp.ccdpars.t_si);
	  cpvec(&startpos,&pos);
	  pos.z = z_image.z;
	  ix=0;
	  do { // do-while lets us insert individual 2d dipoles if npix=0
	    vec dipole2d;
	    //	    cpvec(&z_unit,&dipole2d);
	    cpvec(&dipole_unit,&dipole2d);
	    scalevec(&dipole2d,dpa->dipole_moment*xi_moment);
	    multipole_stack_append(&msi,dipole_2d,&dipole2d,&pos,&sym_axis);
	    vec_add(&pos,&step,&pos);
	  } while (ix++ < dpa->npix);
	}
      }

      // and do the 3d dipoles
      for (dls_ix=0;dls_ix<dlm.ndls;dls_ix++) {
	dipole_list_struct *dls=&dlm.dls[dls_ix];
	vec pos;
	vec half_pixel_offset={0.5,0.5,0};
	//	vec half_pixel_offset={0.5*0.01/10.0,0.5*0.01/10.0,0};
	cpvec(&(dls->p),&pos);
	vec_add(&pos,&half_pixel_offset,&pos);
	scalevec(&pos,0.01/10.0); // now it's in cm
	vec z_unit={0,0,1};
	vec z_image;
	int each;

	for (each=0;each<2;each++) { // each=0 is for the image on the other
	  // side of the channel. each=1 is for the image on the other side
	  // of the backside electrode
	  if ((im==0) && (each==1)) continue;   // avoid counting twice
	  if ((im==nim) && (each==0)) continue; // special attention to field at the backside.
	  cpvec(&z_unit,&z_image);
	  scalevec(&z_image,(float)2*im*(2*each-1)*ccd_rp.ccdpars.t_si);
	  pos.z = z_image.z;
	  vec dipole3d;
	  cpvec(&z_unit,&dipole3d);
	  scalevec(&dipole3d,dls->dipole_moment*p_moment);
	  multipole_stack_append(&msi,dipole_3d,&dipole3d,&pos,NULL);
	}
      }
    }

    //     multipole_stack_inspect(msi);
    // test drift from 0,0
    {
      int i;
      int n_vals=ccd_rp.ccdpars.n_sigma;
      driftstruct ds;
      
      ds.sigma_resp=(double*)malloc(n_vals*sizeof(double));
      ds.tcol=(double*)malloc(n_vals*sizeof(double));
      ds.emag=(double*)malloc(n_vals*sizeof(double));
      ds.potential=(double*)malloc(n_vals*sizeof(double));
      ds.pos=(vec*)malloc(n_vals*sizeof(vec));
      if ((ds.pos == NULL) || 
	  (ds.tcol == NULL) ||
	  (ds.emag == NULL) ||
	  (ds.potential == NULL) ||
	  (ds.sigma_resp == NULL)) {
	fprintf(stderr,"can't allocate. exiting..\n");
	exit(1);
      }

      if (slm.nss > 0) {
	vec ps,pscpy,part1,part2;
	int sl_ix;
	int j=0;
	int iter=0;
	int iss;

	FILE *fp=fopen("eye.tnt","w");
	fprintf(fp,"dat\n");
	fprintf(fp,"slice_ix\t");
	fprintf(fp,"z[um]\t");
	fprintf(fp,"x[um]\t");
	fprintf(fp,"y[um]\t");
	fprintf(fp,"delta x[um]\t");
	fprintf(fp,"delta y[um]\t");
	fprintf(fp,"|E|[v/cm]\t");
	fprintf(fp,"potential [v]\t");
	fprintf(fp,"tcol [s]\t");
	fprintf(fp,"sigma [um]\n");

	for (iss=0;iss<slm.nss;iss++) {
	  for (sl_ix=0;sl_ix<slm.slcs[iss].n_elem;sl_ix++) {
	    // compute the starting point based on a weight between the two 
	    // extremes of the slice (currently expressed in pixels)
	    float wt=1.0;
	    if (slm.slcs[iss].n_elem>1) 
	      wt=sl_ix/((float)slm.slcs[iss].n_elem-1);
	    if (slm.slcs[iss].slice[0].x==0 &&
		slm.slcs[iss].slice[0].y==0 && 0) {
	      slm.slcs[iss].slice[0].x+=1e-36;
	      slm.slcs[iss].slice[1].x+=1e-36;
	      slm.slcs[iss].slice[0].y+=1e-36;
	      slm.slcs[iss].slice[1].y+=1e-36;
	    }
	    cpvec(&slm.slcs[iss].slice[0],&part1);
	    cpvec(&slm.slcs[iss].slice[1],&part2);
	    scalevec(&part1,wt*0.01/10.0);
	    scalevec(&part2,(1.0-wt)*0.01/10.0);
	    vec_add(&part1,&part2,&ps);

	    ps.z=ccd_rp.ccdpars.z[0];
	    cpvec(&ps,&pscpy);

	    drift (&ps,&ds,
		   &ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		   n_vals,FORWARD,&iter);
	    // and report
	    for (i=0;i<=iter;i++) {
	      fprintf(fp,"%d %g %g %g %g %g %g %g %g %g\n",sl_ix,
		      ds.pos[i].z*1e4,ds.pos[i].x*1e4,ds.pos[i].y*1e4,
		      (ds.pos[i].x-pscpy.x)*1e4,(ds.pos[i].y-pscpy.y)*1e4,
		      ds.emag[i],ds.potential[i],ds.tcol[i],
		      ds.sigma_resp[i]);
	    }
	  }
	}
	fclose(fp);
      } 
      if (wid.nwids>0) {
	// compute the widths of each pixel as specified
	int wid_ix;
	int aim_x1,aim_y1;
	int aim_x2,aim_y2;
	int aim_x3,aim_y3;
	float startx,starty;
	FILE *fffp=fopen("pixelwidths.dat","w");

	for (wid_ix=0;wid_ix<wid.nwids;wid_ix++) {
	  int pix_ix,npixstep,this_pix;
	  npixstep=(int)fabs(wid.wids[wid_ix].pix[1]-wid.wids[wid_ix].pix[0])+1;
	  for (pix_ix=0;pix_ix<npixstep;pix_ix++) {
	    this_pix=wid.wids[wid_ix].pix[0]+
	      pix_ix*((wid.wids[wid_ix].pix[1]>wid.wids[wid_ix].pix[0])?1:-1);
	    //	    fprintf(stdout,"this_pix %d\n",this_pix);
	    startx = starty = 0.0;

	    switch(wid.wids[wid_ix].way) {
	    case BND_COLUMN:
	      aim_x1=this_pix-1;
	      aim_x2=aim_x1+1;
	      aim_x3=aim_x1+2;
	      startx=aim_x1*0.01/10.0;
	      aim_y1=(int)floor(wid.wids[wid_ix].transv_pixcoord);
	      aim_y2=aim_y1;
	      aim_y3=aim_y1;
	      starty=wid.wids[wid_ix].transv_pixcoord*0.01/10.0;
	      break;
	    case BND_ROW:
	      aim_x1=(int)floor(wid.wids[wid_ix].transv_pixcoord);
	      aim_x2=aim_x1;
	      aim_x3=aim_x1;
	      startx=wid.wids[wid_ix].transv_pixcoord*0.01/10.0;
	      aim_y1=this_pix-1;
	      aim_y2=aim_y1+1;
	      aim_y3=aim_y1+2;
	      starty=aim_y1*0.01/10.0;
	      break;
	    default:
	      break;
	    }

	    vec ps={startx,starty,ccd_rp.ccdpars.z[0]};
	    vec ps1,ps2,ps3,pinch12,pinch23;
	    cpvec(&ps,&ps1);
	    cpvec(&ps,&ps2);
	    cpvec(&ps,&ps3);
	    
	    target_pix(aim_x1,aim_y1,&ps1,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		       n_vals,wid.wids[wid_ix].way);
	    target_pix(aim_x2,aim_y2,&ps2,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		       n_vals,wid.wids[wid_ix].way);
	    target_pix(aim_x3,aim_y3,&ps3,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		       n_vals,wid.wids[wid_ix].way);

	    vec tv1,tv2;
	    cpvec(&ps1,&tv1);
	    cpvec(&ps2,&tv2);

	    if (multipole_stack_pinch(&tv1,&tv2,&pinch12,
				      wid.wids[wid_ix].way,
				      &ccd_rp.ccdpars,ccd_rp.ccdpars.e,
				      msi) == -1) continue;
	    cpvec(&ps2,&tv1);
	    cpvec(&ps3,&tv2);
	    if (multipole_stack_pinch(&tv1,&tv2,&pinch23,
				      wid.wids[wid_ix].way,
				      &ccd_rp.ccdpars,ccd_rp.ccdpars.e,
				      msi) == -1) continue;

	    // when control arrives here we've pinched both boundaries.
	    //	    fprintf(stderr,"%s\n",show_vector("pinch12",&pinch12));
	    //	    fprintf(stderr,"%s\n",show_vector("pinch23",&pinch23));
	    float width;
	    vec difference,average;

	    vec_diff(&pinch12,&pinch23,&difference);
	    vec_add(&pinch12,&pinch23,&average);
	    scalevec(&average,0.5);
	    width=modulus(&difference)*1e4;
	    //	      fprintf(stderr,"width %f[um]\n",width);
	    
	    switch (wid.wids[wid_ix].way) {
	    case BND_COLUMN:
	      fprintf(fffp,"xpix %d yloc %f xfracwid %f xshift %f\n",
		      this_pix,wid.wids[wid_ix].transv_pixcoord,width/10.0,
		      (average.x/(0.01/10.0)-0.5)-this_pix);
	      break;
	    case BND_ROW:
	      fprintf(fffp,"ypix %d xloc %f yfracwid %f yshift %f\n",
		      this_pix,wid.wids[wid_ix].transv_pixcoord,width/10.0,
		      (average.y/(0.01/10.0)-0.5)-this_pix);
	      break;
	    default:
	      break;
	    }
	  }
	  //	  fprintf(stdout,"xloc %f from %d to %d: %d\n",
	  //	  wid.wids[wid_ix].transv_pixcoord,
	  //	  wid.wids[wid_ix].pix[0],
	  //	  wid.wids[wid_ix].pix[1],
	  //	  wid.wids[wid_ix].way);
	}
	fclose(fffp);
      }

      if (crn.ncrns>0) {
	// compute corner-to-corner boundary locus
	// compute the boundary locus

	int crn_ix;
	int gen_ix;
	FILE *fgp,*ggp;
	if (((fgp=fopen("trace.qdp","w"))==NULL) ||
	    ((ggp=fopen("boundary_distort.txt","w"))==NULL)) {
	  fprintf(stderr,"can't open file ..\nexiting\n");
	  exit(1);
	}

	for (crn_ix=0;crn_ix<crn.ncrns;crn_ix++) {
	  locate_pix_edge(&crn.crns[crn_ix],&ccd_rp,msi);
	  fprintf(ggp,"# EDGE TRACE\n");
	  fprintf(ggp,"x0=%g y0=%g x1=%g y1=%g n=%d\n",
		  crn.crns[crn_ix].corner[0].x,
		  crn.crns[crn_ix].corner[0].y,
		  crn.crns[crn_ix].corner[1].x,
		  crn.crns[crn_ix].corner[1].y,
		  crn.crns[crn_ix].n_elem);
	  
	  for (gen_ix=0;gen_ix<crn.crns[crn_ix].n_elem;gen_ix++) {
	    
	    fprintf(ggp,"%d %g %g\n",
		    gen_ix,
		    crn.crns[crn_ix].dist_along_edge[gen_ix],
		    crn.crns[crn_ix].edge_distortion[gen_ix]-
		    crn.crns[crn_ix].pixcoord[(crn.crns[crn_ix].way==BND_ROW)?1:0]*
		    0.01/10.0);

	    if (crn.crns[crn_ix].way==BND_ROW) {
	      fprintf(fgp,"%15g %15g\n",
		      crn.crns[crn_ix].dist_along_edge[gen_ix],
		      crn.crns[crn_ix].edge_distortion[gen_ix]);
	    } else {
	      fprintf(fgp,"%15g %15g\n",
		      crn.crns[crn_ix].edge_distortion[gen_ix],
		      crn.crns[crn_ix].dist_along_edge[gen_ix]);
	    }
	  }
	}
	fclose(fgp);
	fclose(ggp);
	exit(0);
	/* FILE *fgp=fopen("sora.tnt","w"); */

	/* if (old_style) { */
	/*   fprintf(fgp,"data\n"); */
	/*   fprintf(fgp,"xpix\t"); */
	/*   fprintf(fgp,"ypix\t"); */
	/*   fprintf(fgp,"xbnd[um]\t"); */
	/*   fprintf(fgp,"ybnd[um]\n"); */
	/* } else { */
	/*   fprintf(fgp,"data\n"); */
	/*   fprintf(fgp,"xpix\t"); */
	/*   fprintf(fgp,"ypix\t"); */
	/*   fprintf(fgp,"bnd_type\t"); */
	/*   fprintf(fgp,"slice_ix\t"); */
	/*   fprintf(fgp,"z[um]\t"); */
	/*   fprintf(fgp,"xbnd[um]\t"); */
	/*   fprintf(fgp,"ybnd[um]\t"); */
	/*   fprintf(fgp,"delta x [um]\t"); */
	/*   fprintf(fgp,"delta y [um]\t"); */
	/*   fprintf(fgp,"|E|[v/cm]\t"); */
	/*   fprintf(fgp,"potential [v]\t"); */
	/*   fprintf(fgp,"tcol [s]\t"); */
	/*   fprintf(fgp,"sigma [um]\n"); */
	/* } */

	/* for (crn_ix=0;crn_ix<crn.ncrns;crn_ix++) { */
	/*   for (aim_ix=0;aim_ix<crn.crns[crn_ix].n_elem;aim_ix++) { */
	/*     float t=aim_ix/((float)crn.crns[crn_ix].n_elem-1); */
	/*     startx = starty = 0.0; */
	/*     if (crn.crns[crn_ix].way == BND_COLUMN) { */
	/*       //	      aim_x1=crn.crns[crn_ix].pix; */
	/*       aim_x1=0; // dummy */
	/*       aim_x2=aim_x1+1; */
	/*       startx=aim_x1*0.01/10.0; */
	/*       //	      aim_y1=(int)floor(crn.crns[crn_ix].limits[0]); */
	/*       aim_y1=0; //dummy */
	/*       aim_y2=aim_y1; */
	/*       //	      starty=((1-t)*crn.crns[crn_ix].limits[0]+ */
	/*       //		      t*crn.crns[crn_ix].limits[1])*0.01/10.0; */
	/*     } else { */
	/*       // crn.way is equal to BND_ROW */
	/*       //	      aim_x1=(int)floor(crn.crns[crn_ix].limits[0]); */
	/*       aim_x1=0; //dummy */
	/*       aim_x2=aim_x1; */
	/*       //	      startx=((1-t)*crn.crns[crn_ix].limits[0]+ */
	/*       //		      t*crn.crns[crn_ix].limits[1])*0.01/10.0; */
	/*       //	      aim_y1=crn.crns[crn_ix].pix; */
	/*       aim_y1=0; //dummy */
	/*       aim_y2=aim_y1+1; */
	/*       starty=aim_y1*0.01/10.0; */
	/*     } */

	/*     vec ps={startx,starty,ccd_rp.ccdpars.z[0]}; */
	/*     vec ps1,ps2,pinch; */

	/*     cpvec(&ps,&ps1); */
	/*     cpvec(&ps,&ps2); */

	/*     target_pix(aim_x1,aim_y1,&ps1,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e, */
	/* 	       n_vals,crn.crns[crn_ix].way); */
	/*     target_pix(aim_x2,aim_y2,&ps2,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e, */
	/* 	       n_vals,crn.crns[crn_ix].way); */

	/*     if (multipole_stack_pinch(&ps1,&ps2,&pinch, */
	/* 			      crn.crns[crn_ix].way, */
	/* 			      &ccd_rp.ccdpars,ccd_rp.ccdpars.e, */
	/* 			      msi) == -1) continue; */

	/*     // finally determine which pixel along the other direction  */
	/*     // this position maps to */
	/*     int xf,yf; */
	/*     drift2pix(&pinch,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,n_vals,&xf,&yf); */
	/*     if (crn.crns[crn_ix].way == BND_COLUMN) { */
	/*       xf = aim_x2; */
	/*     } else { */
	/*       yf = aim_y2; */
	/*     } */
	/*     if (old_style) { */
	/*       fprintf(fgp,"%d %d %g %g\n",xf,yf,pinch.x*1e4,pinch.y*1e4); */
	/*     } else { */
	/*       // will output the entire ds trace */
	/*       driftstruct ds; */
	/*       vec pscpy; */
	/*       int my_iter=0; */

	/*       // allocate driftstruct */

	/*       ds.sigma_resp=(double*)malloc(n_vals*sizeof(double)); */
	/*       ds.tcol=(double*)malloc(n_vals*sizeof(double)); */
	/*       ds.emag=(double*)malloc(n_vals*sizeof(double)); */
	/*       ds.potential=(double*)malloc(n_vals*sizeof(double)); */
	/*       ds.pos=(vec*)malloc(n_vals*sizeof(vec)); */
	/*       if ((ds.pos == NULL) ||  */
	/* 	  (ds.tcol == NULL) || */
	/* 	  (ds.emag == NULL) || */
	/* 	  (ds.potential == NULL) || */
	/* 	  (ds.sigma_resp == NULL)) { */
	/* 	fprintf(stderr,"can't allocate. exiting..\n"); */
	/* 	exit(1); */
	/*       } */

	/*       cpvec(&pinch,&pscpy); */
	/*       drift(&pinch,&ds,&ccd_rp.ccdpars,msi, */
	/* 	    ccd_rp.ccdpars.e,n_vals,FORWARD,&my_iter); */
	/*       // printout drift trace */
	/*       { */
	/* 	int i; */
	/* 	for (i=0;i<=my_iter;i++) { */
	/* 	  fprintf(fgp,"%d %d %d %d %g %g %g %g %g %g %g %g %g\n", */
	/* 		  xf,yf,crn.crns[crn_ix].way,i, */
	/* 		  ds.pos[i].z*1e4,ds.pos[i].x*1e4,ds.pos[i].y*1e4, */
	/* 		  (ds.pos[i].x-pscpy.x)*1e4, */
	/* 		  (ds.pos[i].y-pscpy.y)*1e4, */
	/* 		  ds.emag[i],ds.potential[i],ds.tcol[i], */
	/* 		  ds.sigma_resp[i]); */
	/* 	} */
	/*       } */
	/*       // free allocated driftstruct */
	/*       free(ds.sigma_resp); */
	/*       free(ds.tcol); */
	/*       free(ds.emag); */
	/*       free(ds.potential); */
	/*       free(ds.pos); */
	/*     } */
	/*   } */
	/* } */
	/* fclose(fgp); */
      }

      if (bnd.nbnds>0) {
	// compute the boundary locus
	int aim_x1,aim_y1;
	int aim_x2,aim_y2;
	float startx,starty;
	int aim_ix;
	int bnd_ix;
	int old_style=1;

	FILE *ffp=fopen("sore.tnt","w");

	if (old_style) {
	  fprintf(ffp,"data\n");
	  fprintf(ffp,"xpix\t");
	  fprintf(ffp,"ypix\t");
	  fprintf(ffp,"xbnd[um]\t");
	  fprintf(ffp,"ybnd[um]\n");
	} else {
	  fprintf(ffp,"data\n");
	  fprintf(ffp,"xpix\t");
	  fprintf(ffp,"ypix\t");
	  fprintf(ffp,"bnd_type\t");
	  fprintf(ffp,"slice_ix\t");
	  fprintf(ffp,"z[um]\t");
	  fprintf(ffp,"xbnd[um]\t");
	  fprintf(ffp,"ybnd[um]\t");
	  fprintf(ffp,"delta x [um]\t");
	  fprintf(ffp,"delta y [um]\t");
	  fprintf(ffp,"|E|[v/cm]\t");
	  fprintf(ffp,"potential [v]\t");
	  fprintf(ffp,"tcol [s]\t");
	  fprintf(ffp,"sigma [um]\n");
	}

	for (bnd_ix=0;bnd_ix<bnd.nbnds;bnd_ix++) {
	  for (aim_ix=0;aim_ix<bnd.bnds[bnd_ix].n_elem;aim_ix++) {
	    float t=aim_ix/((float)bnd.bnds[bnd_ix].n_elem-1);
	    startx = starty = 0.0;
	    if (bnd.bnds[bnd_ix].way == BND_COLUMN) {
	      aim_x1=bnd.bnds[bnd_ix].pix;
	      aim_x2=aim_x1+1;
	      startx=aim_x1*0.01/10.0;
	      aim_y1=(int)floor(bnd.bnds[bnd_ix].limits[0]);
	      aim_y2=aim_y1;
	      starty=((1-t)*bnd.bnds[bnd_ix].limits[0]+
		      t*bnd.bnds[bnd_ix].limits[1])*0.01/10.0;
	    } else {
	      // bnd.way is equal to BND_ROW
	      aim_x1=(int)floor(bnd.bnds[bnd_ix].limits[0]);
	      aim_x2=aim_x1;
	      startx=((1-t)*bnd.bnds[bnd_ix].limits[0]+
		      t*bnd.bnds[bnd_ix].limits[1])*0.01/10.0;
	      aim_y1=bnd.bnds[bnd_ix].pix;
	      aim_y2=aim_y1+1;
	      starty=aim_y1*0.01/10.0;
	    }

	    vec ps={startx,starty,ccd_rp.ccdpars.z[0]};
	    vec ps1,ps2,pinch;

	    cpvec(&ps,&ps1);
	    cpvec(&ps,&ps2);

	    target_pix(aim_x1,aim_y1,&ps1,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		       n_vals,bnd.bnds[bnd_ix].way);
	    target_pix(aim_x2,aim_y2,&ps2,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
		       n_vals,bnd.bnds[bnd_ix].way);

	    if (multipole_stack_pinch(&ps1,&ps2,&pinch,
				      bnd.bnds[bnd_ix].way,
				      &ccd_rp.ccdpars,ccd_rp.ccdpars.e,
				      msi) == -1) continue;

	    // finally determine which pixel along the other direction 
	    // this position maps to
	    int xf,yf;
	    drift2pix(&pinch,&ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,n_vals,&xf,&yf);
	    if (bnd.bnds[bnd_ix].way == BND_COLUMN) {
	      xf = aim_x2;
	    } else {
	      yf = aim_y2;
	    }
	    if (old_style) {
	      fprintf(ffp,"%d %d %g %g\n",xf,yf,pinch.x*1e4,pinch.y*1e4);
	    } else {
	      // will output the entire ds trace
	      driftstruct ds;
	      vec pscpy;
	      int my_iter=0;

	      // allocate driftstruct

	      ds.sigma_resp=(double*)malloc(n_vals*sizeof(double));
	      ds.tcol=(double*)malloc(n_vals*sizeof(double));
	      ds.emag=(double*)malloc(n_vals*sizeof(double));
	      ds.potential=(double*)malloc(n_vals*sizeof(double));
	      ds.pos=(vec*)malloc(n_vals*sizeof(vec));
	      if ((ds.pos == NULL) || 
		  (ds.tcol == NULL) ||
		  (ds.emag == NULL) ||
		  (ds.potential == NULL) ||
		  (ds.sigma_resp == NULL)) {
		fprintf(stderr,"can't allocate. exiting..\n");
		exit(1);
	      }

	      cpvec(&pinch,&pscpy);
	      drift(&pinch,&ds,&ccd_rp.ccdpars,msi,
		    ccd_rp.ccdpars.e,n_vals,FORWARD,&my_iter);
	      // printout drift trace
	      {
		int i;
		for (i=0;i<=my_iter;i++) {
		  fprintf(ffp,"%d %d %d %d %g %g %g %g %g %g %g %g %g\n",
			  xf,yf,bnd.bnds[bnd_ix].way,i,
			  ds.pos[i].z*1e4,ds.pos[i].x*1e4,ds.pos[i].y*1e4,
			  (ds.pos[i].x-pscpy.x)*1e4,
			  (ds.pos[i].y-pscpy.y)*1e4,
			  ds.emag[i],ds.potential[i],ds.tcol[i],
			  ds.sigma_resp[i]);
		}
	      }
	      // free allocated driftstruct
	      free(ds.sigma_resp);
	      free(ds.tcol);
	      free(ds.emag);
	      free(ds.potential);
	      free(ds.pos);
	    }
	  }
	}
	fclose(ffp);
      }

      if (0) {
	vec ps={0,0,0};

	ps.x=0;
	ps.y=0;
	ps.z=ccd_rp.ccdpars.z[0];

	int iter=0;

	drift (&ps,&ds,
	       &ccd_rp.ccdpars,msi,ccd_rp.ccdpars.e,
	       n_vals,FORWARD,&iter);
	{
	  int j=0;
	  int i;
	  FILE *fp=stdout;

	  for (i=0;i<=iter;i++) {
	    fprintf(fp,"%d %g %g %g %g %g %g %g %g\n",i,
		    ds.pos[i].z,ds.pos[i].x,ds.pos[i].y,
		    ps.x,ds.emag[i],ds.potential[i],ds.tcol[i],
		    ds.sigma_resp[i]);
	  }
	}
      }
      //    cleanup:
      if (0) { 
	// for some reason freeing up allocated arrays was triggering
	// a core dump (!?) skip for now, end of program.
	free(ds.sigma_resp);
	free(ds.tcol);
	free(ds.emag);
	free(ds.potential);
	free(ds.pos);
      }
    }
  }
  exit(0);
}

void dipole_3d (vec *x_vec,vec *p_vec,vec *x0_vec,vec *dummy,vec *field) {
  vec tmp_n_vec;
  vec tmp_r_vec;
  double r;
  if (x0_vec==NULL) {
    cpvec(x_vec,&tmp_r_vec);
  } else {
    vec_diff(x_vec,x0_vec,&tmp_r_vec);
  }
  r=sqrt(dot_prod(&tmp_r_vec,&tmp_r_vec));
  cpvec(&tmp_r_vec,&tmp_n_vec);
  unitvec(&tmp_n_vec);
  scalevec(&tmp_n_vec,3*dot_prod(p_vec,&tmp_n_vec));
  vec_diff(&tmp_n_vec,p_vec,field);
  scalevec(field,1.0/(4*M_PI*e_0*e_si*pow(r,3)));
  return;
}

void dipolegrid_2d (vec *x_vec,vec *p_vec,vec 
		    *x0_vec,vec *sym_axis,vec *field) {
  vec p_uvec;
  vec sym_uvec;
  vec lat_vec;
  vec r_vec;

  // this is for an infinite grid of dipoles periodic in the direction 
  // perpendicular to both sym_axis and also to z. p_vec may be rotated in 
  // the plane that is perpendicular to sym_axis. this has to be handled above.

  // sanity check: sym_axis should be orthogonal to p_vec
  cpvec(p_vec,&p_uvec);  
  unitvec(&p_uvec);
  cpvec(sym_axis,&sym_uvec); 
  unitvec(&sym_uvec);

  double dp = dot_prod(&sym_uvec,&p_uvec);
  if (dp*dp>1e-8) {
    fprintf(stderr,"dot product between symmetry axis and dipole moment is not zero!\nexiting..\n");
    exit(1);
  }

  // establish the lateral coordinate by lateral = sym_axis x p_uvec and then remove any z component.
  vec lat_uvec;
  cross_prod(&sym_uvec,&p_uvec,&lat_uvec);
  lat_uvec.z=0;
  unitvec(&lat_uvec);
  // decompose p_vec into lat & z components
  vec z_uvec={0,0,1};
  double p_lat,p_z;
  p_lat=dot_prod(&lat_uvec,p_vec);
  p_z  =dot_prod(&z_uvec,p_vec);

  vec vA,vB,tmp;
  cpvec(&lat_uvec,&vA);
  scalevec(&vA,p_lat);
  cpvec(&z_uvec,&tmp);
  scalevec(&tmp,-p_z);
  vec_add(&vA,&tmp,&vA);

  cpvec(&z_uvec,&vB);
  scalevec(&vB,p_lat);
  cpvec(&lat_uvec,&tmp);
  scalevec(&tmp,p_z);
  vec_add(&vB,&tmp,&vB);

  if (x0_vec==NULL) {
    cpvec(x_vec,&r_vec);
  } else {
    vec_diff(x_vec,x0_vec,&r_vec);
  }
  
  double rho_lat,rho_z;
  rho_lat=dot_prod(&r_vec,&lat_uvec);
  rho_z  =dot_prod(&r_vec,&z_uvec);

  double a,b;
  double pixel=0.01/10.0;
  a = 2*M_PI*rho_lat/pixel;
  b = 2*M_PI*rho_z/pixel;

  double coshb=cosh(b);
  double cosa=cos(a);
  double A,B;
  
  A=(pow(coshb,-2)-cosa/coshb)/pow(1-cosa/coshb,2);
  B=sin(a)*tanh(b)/coshb/pow(1-cosa/coshb,2);

  scalevec(&vA,A);
  scalevec(&vB,B);
  vec_add(&vA,&vB,field);
  scalevec(field,M_PI/(e_0*e_si*pow(pixel,2)));
  return;
}

void dipolegrid_2d_orig (vec *x_vec,vec *p_vec,vec 
		    *x0_vec,vec *sym_axis,vec *field) {
  vec p_uvec;
  vec lat_vec;
  vec lat_uvec;
  vec r_vec;

  // this is for an infinite grid of dipoles periodic in the direction 
  // perpendicular to both sym_axis and p_vec.

  cpvec(p_vec,&p_uvec);
  unitvec(&p_uvec);

  if (x0_vec==NULL) {
    cpvec(x_vec,&r_vec);
  } else {
    vec_diff(x_vec,x0_vec,&r_vec);
  }

  if (sym_axis != NULL) {
    // make use of the axis of symmetry
    vec sym;
    cpvec(sym_axis,&sym);
    unitvec(&sym);
    // subtract from r_vec any component in sym
    scalevec(&sym,dot_prod(&sym,&r_vec));
    vec_diff(&r_vec,&sym,&lat_vec);
    // remove any component in z
    lat_vec.z=0.0;
    cpvec(&lat_vec,&lat_uvec);
    unitvec(&lat_uvec);
    // done..
  }  
  // field components are in the z & direction perpendicular 
  // to symmetry axis
  double a,b;
  double pixel=0.01/10.0;
  double x0=sqrt(dot_prod(&lat_vec,&lat_vec));
  double z0=r_vec.z;
  double xi=sqrt(dot_prod(p_vec,p_vec));
  double lat_scalar,p_scalar;

  a = 2*M_PI*x0/pixel;
  b = 2*M_PI*z0/pixel;

  double coshb=cosh(b);

  lat_scalar = sin(a)*tanh(b)/coshb/pow(1-cos(a)/coshb,2);
  p_scalar   = (cos(a)/coshb-pow(coshb,-2))/pow(1-cos(a)/coshb,2);

  scalevec(&lat_uvec, lat_scalar);
  scalevec(&p_uvec,p_scalar);

  vec_add(&lat_uvec,&p_uvec,field);

  scalevec(field,M_PI*xi/(e_0*e_si*pow(pixel,2)));
  //  fprintf(stderr,"(a,b)=(%f,%f) - %s\n",a,b,show_vector("field",field));
  return;
}

void dipole_2d (vec *x_vec,vec *p_vec,vec *x0_vec,
		vec *sym_axis,vec *field) {
  vec tmp_n_vec;
  vec tmp_r_vec;
  double r;

  if (x0_vec==NULL) {
    cpvec(x_vec,&tmp_r_vec);
  } else {
    vec_diff(x_vec,x0_vec,&tmp_r_vec);
  }
  if (sym_axis != NULL) {
    // make use of the axis of symmetry
    vec sym;
    cpvec(sym_axis,&sym);
    unitvec(&sym);
    // subtract from tmp_r_vec any component in sym
    scalevec(&sym,dot_prod(&sym,&tmp_r_vec));
    vec_diff(&tmp_r_vec,&sym,&tmp_r_vec);
    // done..
  }  

  r=sqrt(dot_prod(&tmp_r_vec,&tmp_r_vec));
  cpvec(&tmp_r_vec,&tmp_n_vec);
  unitvec(&tmp_n_vec);
  scalevec(&tmp_n_vec,2*dot_prod(p_vec,&tmp_n_vec));
  vec_diff(&tmp_n_vec,p_vec,field);
  scalevec(field,1.0/(2*M_PI*e_0*e_si*pow(r,2)));
  return;
}

void trunc_dipole_2d (vec *x_vec,vec *p_vec,vec *x0_vec,
		      vec *trunc_axis,vec *field) {
  // trunc_axis is the vector that points away from the truncation point
  // toward infinity, according to the specified phi value. The opposite direction contains 
  // the distributed line dipole. This truncation axis is used similarly to the
  // symmetry axis for the non-truncated dipole.
  // if phi=0, trunc_axis is parallel to \hat{y}
  // if phi=90, trunc_axis is parallel to \hat{x}
  //
  // derived expression requires (2d) rho, (3d) r and various
  // dot products
  vec r_vec;
  vec r_uvec;
  vec rho_vec;
  vec rho_uvec;
  vec trunc_uvec;
  vec part1,part2;
  double cos_theta0;
  double sin_theta0;
  double rho_dot_xi;
  double rho;
  
  // prepare r_vec
  if (x0_vec==NULL) {
    cpvec(x_vec,&r_vec);
  } else {
    vec_diff(x_vec,x0_vec,&r_vec);
  }

  // prepare r_uvec
  cpvec(&r_vec,&r_uvec);
  unitvec(&r_uvec);

  // prepare trunc_uvec
  cpvec(trunc_axis,&trunc_uvec);
  unitvec(&trunc_uvec);

  // prepare cos_theta0 & sin_theta0
  cos_theta0=dot_prod(&r_uvec,&trunc_uvec);
  sin_theta0=sin(1-pow(cos_theta0,2));
  
  // prepare rho_vec
  cpvec(trunc_axis,&rho_vec);
  scalevec(&rho_vec,-1*dot_prod(&r_vec,&trunc_uvec));
  vec_add(&r_vec,&rho_vec,&rho_vec);
  rho=modulus(&rho_vec);

  // prepare rho_uvec
  cpvec(&rho_vec,&rho_uvec);
  unitvec(&rho_uvec);
  
  // prepare rho_uvec_dot_xi
  rho_dot_xi = dot_prod(&rho_uvec,p_vec);

  // prepare part1
  cpvec(&rho_uvec,&part1);
  scalevec(&part1,rho_dot_xi*(2-cos_theta0*(3-cos_theta0*cos_theta0)));

  // prepare part2
  cpvec(trunc_axis,&part2);
  scalevec(&part2,rho_dot_xi*pow(sin_theta0,3));

  // collect parts 1 & 2 into part1
  vec_add(&part1,&part2,&part1);

  // prepare part3 but store in part2
  cpvec(p_vec,&part2);
  scalevec(&part2,cos_theta0-1.0);

  // collect part3 in to part1
  vec_add(&part1,&part2,&part1);
  
  // perform scaling then shipout
  scalevec(&part1,1.0/(4*M_PI*e_0*e_si*pow(rho,2)));
  cpvec(&part1,field); 
  return;
}

void drift (vec *ps,driftstruct *ds,
 	    bi_ccd_pars *pars,multipole_stack *msi,double *e,
	    int n_vals,enum direction dir,int *iter){

  // routine populates the driftstruct *ds given the initial position *ps
  // bi_ccd_pars *pars, the multipole_stack *msi and the computed field 
  // z component *e. arrays trying to encapsulate..
  
  // if dir=FORWARD then assume we are following a test particle from the 
  // location at the channel *ps toward the backside surface. reverse the 
  // electric field vector if its z component is negative.
  vec null_field={0,0,0};
  *iter=0;
  float dz=2*(-pars->t_si)/(pars->n_sigma-1);
  float emag;
  float potential=0;
  float tcoll=0;
  // if dir==REVERSE increment ps->z from its current value until
  // the superposed field has a negative component in y. from there
  // trace the field backward until the vector integral intersects pars->t_si
  vec drift_field;

  if (dir==BACKWARD) {
    // modifying ps->z only
    if (ps->z <= 0) {
      ps->z=-dz;
    } else {
      // otherwise use the current value of ps->z
    }
    do {
      int i=floor((ps->z - pars->t_si)/dz);
      if (i>=n_vals) i=n_vals-1;
      if (i<1)       i=1;
      vec field;
      cpvec(&null_field,&field);
      fprintf(stderr,"entering mutlipole_stack_eval: %s\n",show_vector("ps=",ps));
      multipole_stack_eval(msi,ps,&field);
      fprintf(stderr,"exiting mutlipole_stack_eval: %s\n",show_vector("field=",&field));
      cpvec(&null_field,&drift_field);
      drift_field.z=0.5*(e[i-1]+e[i]);
      vec_add(&field,&drift_field,&drift_field);
      //      if (drift_field.z>0) ps->z -= dz;
    } while (drift_field.z>0);
  }

  // ps->x and/or ps->y become nan within the next several lines..

  do {
    int i=floor((ps->z - pars->t_si)/dz);
    if (i>=n_vals) i=n_vals-1;
    if (i<1)       i=1;
    vec field;
    cpvec(&null_field,&field);

    multipole_stack_eval(msi,ps,&field);

    cpvec(&null_field,&drift_field);
    drift_field.z=0.5*(e[i-1]+e[i]);

    vec_add(&field,&drift_field,&drift_field);

    emag=modulus(&drift_field);

    if (dir==BACKWARD) {
      potential -= dz*emag;
      ds->emag[*iter]=emag;
      ds->potential[*iter]=potential;
      tcoll-=fabs(dz/(emag*mu_Si(emag,pars->T)));
      ds->tcol[*iter]=tcoll;
      unitvec(&drift_field);
      scalevec(&drift_field,+dz);
    } else {
      potential += dz*emag;
      ds->emag[*iter]=emag;
      ds->potential[*iter]=potential;
      tcoll+=fabs(dz/(emag*mu_Si(emag,pars->T)));
      ds->tcol[*iter]=tcoll;
      unitvec(&drift_field);
      scalevec(&drift_field,-dz);
    }

    vec_add(ps,&drift_field,ps);

    cpvec(ps,&(ds->pos[*iter]));
    (*iter)++;
  } while ((ps->z > 5e-5) && (ps->z < pars->t_si) && 
	   (*iter<2*n_vals-1) && (emag>200)); // here

  {
    int i=*iter;
    float mob0=sqrt(2*k*pars->T/q*mu_Si(0,pars->T))*1e4;

    if (dir==BACKWARD) tcoll=0;
    while (i--) {
      ds->tcol[i] = (tcoll - ds->tcol[i]);
      ds->sigma_resp[i] = mob0*sqrt(ds->tcol[i]);
    } 
  }
  (*iter)--;
}

int multipole_stack_pinch(vec *p1,vec *p2,vec *pinch,
			  enum boundarytype bndtype,
			  bi_ccd_pars *pars,double *e1,
			  multipole_stack *msi) {
  // rewrite of multipole_stack_locate_saddle 
  // to do a pure forward fold
  int i,retval=0;
  int x[3],y[3],*xp[3],*yp[3];
  int n_vals=N_LAYERS;
  int swapix;
  vec *p[3];
  float span;

  for (i=0;i<3;i++) {    xp[i]=&x[i];    yp[i]=&y[i];  }

  p[0]=p1;
  p[1]=p2;
  p[2]=pinch;

  vec_add(p[0],p[1],p[2]);
  scalevec(p[2],0.5);

  for (i=0;i<3;i++) {
    drift2pix(p[i],pars,msi,e1,n_vals,xp[i],yp[i]);
  }

  do {
    if ((*xp[0] == *xp[1]) && (*yp[0] == *yp[1])) {
      retval=1;
      goto abandon;
    }
    if (bndtype==BND_EITHER) {
      if ((*xp[0] == *xp[2]) && (*yp[0] == *yp[2])) {
	swapix=0;
      } else {
	if ((*xp[1] == *xp[2]) && (*yp[1] == *yp[2])) {
	  swapix=1;
	} else {
	  retval=1;
	  goto abandon;
	}
      }
    }
    if (bndtype==BND_COLUMN) {
      if (*xp[0] == *xp[1]) {
	retval=1;
	goto abandon;
      }
      if (*xp[0] == *xp[2]) {
	swapix=0;
      } else {
	if (*xp[1] == *xp[2]) {
	  swapix=1;
	} else {
	  retval=1;
	  goto abandon;
	}
      }
    }
    if (bndtype==BND_ROW) {
      if (*yp[0] == *yp[1]) {
	retval=1;
	goto abandon;
      }
      if (*yp[0] == *yp[2]) {
	swapix=0;
      } else {
	if (*yp[1] == *yp[2]) {
	  swapix=1;
	} else {
	  retval=1;
	  goto abandon;
	}
      }
    }
    {
      void *tmp;
      tmp=p[swapix];      p[swapix] =p[2];     p[2] =tmp;
      tmp=xp[swapix];     xp[swapix]=xp[2];    xp[2]=tmp;
      tmp=yp[swapix];     yp[swapix]=yp[2];    yp[2]=tmp;
    }
    // compute the new p[2], *xp[2] & *yp[2].
    vec_add(p[0],p[1],p[2]);
    scalevec(p[2],0.5);
    drift2pix(p[2],pars,msi,e1,n_vals,xp[2],yp[2]);
    {
      vec tmpo;
      vec_diff(p[0],p[1],&tmpo);
      span=modulus(&tmpo);
    }
  } while (span>0.001e-5); // cm
  cpvec(p[2],pinch);
 abandon:
  return(retval);
}

void target_pix(int x,int y,vec *ps,bi_ccd_pars *pars,multipole_stack *msi,
		double *e,int n_vals,enum boundarytype bndtype) {
  // takes in pixel values x & y and returns in *ps a position that provides 
  // this output (via drift)
  int xpix,ypix;
  float xscal,yscal;
  //  fprintf(stderr,"target_pix1\n");
  drift2pix (ps,pars,msi,e,n_vals,&xpix,&ypix);
  //  fprintf(stderr,"target_pix2\n");
  int it=0;
  int retry=0;
  xscal=1.0;
  yscal=1.0;
  do {
    if (abs(xpix-x)>=7) {
      xscal=1.0;
    } else {
      xscal *= 0.63;
      if (it/10>1) {
	xscal = 1;
	retry++;
	it=0;
      }
    }
    if (abs(ypix-y)>=7) {
      yscal=1.0;
    } else {
      yscal *= 0.63;
      if (it/10>1) {
	xscal = 1;
	retry++;
	it=0;
      }
    }
    if ((bndtype==BND_EITHER) || (bndtype==BND_COLUMN))
      ps->x -= xscal*(xpix-x)*0.010/10.0;
    if ((bndtype==BND_EITHER) || (bndtype==BND_ROW))
      ps->y -= yscal*(ypix-y)*0.010/10.0;

    drift2pix (ps,pars,msi,e,n_vals,&xpix,&ypix);
    it++;
  } while ((((bndtype==BND_EITHER) && ((abs(xpix-x)>0) || (abs(ypix-y)>0))) ||
	    ((bndtype==BND_COLUMN) && (abs(xpix-x)>0)) || 
	    ((bndtype==BND_ROW) && (abs(ypix-y)>0))) && 
	   (retry<5));

/* (((((bndtype==BND_EITHER) ||  */
/* 	      (bndtype==BND_COLUMN)) &&  */
/* 	     (abs(xpix-x)>0)) ||  */
/* 	    (((bndtype==BND_EITHER) ||  */
/* 	      (bndtype==BND_ROW)) &&  */
/* 	     (abs(ypix-y)>0)))  */
/* 	   && (retry<5)); */

  if (retry==5) {
    // never converged. give up. return -999,-999.
    ps->x = ps->y = -999.0;
  } else {
    //    fprintf(stderr,"aiming for x,y=(%d,%d)\n",x,y);
    //    fprintf(stderr,"xpix,ypix=(%d,%d)\n",xpix,ypix);
    //    fprintf(stderr,"looks like we've got it..\n");
    //    fprintf(stderr,"%s gives (x,y)=(%d,%d)\n",show_vector("pos",ps),xpix,ypix);
  }
}

void drift2pix (vec *ps,bi_ccd_pars *pars,multipole_stack *msi,
		double *e,int n_vals,int *xpix,int *ypix) {
  int iter=0;
  static driftstruct ds;
  float offset=-0.005/10.0;
  offset=0;
  vec pscpy;
  static int inited=0;


  cpvec(ps,&pscpy);

  if (0) {
    ds.sigma_resp=(double*)malloc(n_vals*sizeof(double));
    ds.tcol=(double*)malloc(n_vals*sizeof(double));
    ds.emag=(double*)malloc(n_vals*sizeof(double));
    ds.potential=(double*)malloc(n_vals*sizeof(double));
    ds.pos=(vec*)malloc(n_vals*sizeof(vec));
    if ((ds.pos == NULL) || (ds.tcol == NULL) || (ds.emag == NULL) ||
	(ds.potential == NULL) || (ds.sigma_resp == NULL)) {
      fprintf(stderr,"can't allocate. exiting..\n");
      exit(1);
    }
  } else {
    if (!inited) {
      ds.sigma_resp=(double*)malloc(n_vals*sizeof(double));
      ds.tcol=(double*)malloc(n_vals*sizeof(double));
      ds.emag=(double*)malloc(n_vals*sizeof(double));
      ds.potential=(double*)malloc(n_vals*sizeof(double));
      ds.pos=(vec*)malloc(n_vals*sizeof(vec));
      inited=1;
      if ((ds.pos == NULL) || (ds.tcol == NULL) || (ds.emag == NULL) ||
	  (ds.potential == NULL) || (ds.sigma_resp == NULL)) {
	fprintf(stderr,"can't allocate. exiting..\n");
	exit(1);
      }
    }
  }

  
  drift(&pscpy,&ds,pars,msi,e,n_vals,FORWARD,&iter);

  *xpix=(int)floor((ds.pos[iter-1].x-offset)/(0.01/10.0));
  *ypix=(int)floor((ds.pos[iter-1].y-offset)/(0.01/10.0));

  if (0) {
    free(ds.sigma_resp);
    free(ds.tcol);
    free(ds.emag);
    free(ds.potential);
    free(ds.pos);
  }
}

void locate_pix_edge (crn_subset *crn,ccd_runpars *ccd_rpp,multipole_stack *msi) {
  int aim_x1,aim_y1;
  int aim_x2,aim_y2;
  float startx,starty;
  int aim_ix;
  int old_style=0;
  int n_vals=ccd_rpp->ccdpars.n_sigma;
  vec mean_crn[2];
  
  // locate the corner before computing anything 
  // (this should be moved into the loop below)

  fprintf(stderr,"for pixel (x,y)=(%d,%d) in %d steps:\n",
	  crn->pixcoord[0],crn->pixcoord[1],crn->n_elem);

  aim_x1=crn->pixcoord[0];
  aim_y1=crn->pixcoord[1];
  aim_x2=(crn->way==BND_ROW)?aim_x1+1:aim_x1;
  aim_y2=(crn->way==BND_COLUMN)?aim_y1+1:aim_y1;
  // corner 1: aim_x1,aim_y1 is the coordinate that divides
  // charge channels (x,y)=(aim_x1-1,aim_y1-1) from 
  // (aim_x1-1,aim_y1), (aim_x1,aim_y1-1) and (aim_x1,aim_y1) etc.
  int vertex;
  int aim_x[]={aim_x1,aim_x2};
  int aim_y[]={aim_y1,aim_y2};

  for (vertex=0;vertex<2;vertex++) {

    mean_crn[vertex].x=mean_crn[vertex].y=mean_crn[vertex].z=0;

    vec psstart={(aim_x[vertex]-1)*0.01/10.0,
		 (aim_y[vertex]-1)*0.01/10.0,
		 ccd_rpp->ccdpars.z[0]};

    vec ps[5];
    int x[5],y[5];

    cpvec(&psstart,&ps[0]);    cpvec(&psstart,&ps[1]);    cpvec(&psstart,&ps[2]);
    cpvec(&psstart,&ps[3]);    cpvec(&psstart,&ps[4]);

    // find points at the back side that get mapped to each pixel bounding the corner
    target_pix(aim_x[vertex]-1,aim_y[vertex]-1,&ps[0],
	       &ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,BND_EITHER);
    target_pix(aim_x[vertex],aim_y[vertex]-1,&ps[1],
	       &ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,BND_EITHER);
    target_pix(aim_x[vertex]-1,aim_y[vertex],&ps[2],
	       &ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,BND_EITHER);
    target_pix(aim_x[vertex],aim_y[vertex],&ps[3],
	       &ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,BND_EITHER);

    // now fold forward and identify the pixels into which each starting
    // position drifts toward.

    // verify that each backside position ps[0..4] indeed maps to different pixels

    drift2pix(&ps[0],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,&x[0],&y[0]);	
    drift2pix(&ps[1],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,&x[1],&y[1]);	
    drift2pix(&ps[2],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,&x[2],&y[2]);	
    drift2pix(&ps[3],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,&x[3],&y[3]);
	
    fprintf(stderr,"0: (xpix,ypix)=(%d,%d)\n",x[0],y[0]);
    fprintf(stderr,"1: (xpix,ypix)=(%d,%d)\n",x[1],y[1]);
    fprintf(stderr,"2: (xpix,ypix)=(%d,%d)\n",x[2],y[2]);
    fprintf(stderr,"3: (xpix,ypix)=(%d,%d)\n",x[3],y[3]);

    double min_rms=1e-3;
    double theta,sM,sm,dM,dm;
    theta=0;
    sM = sm = min_rms/sqrt(2.0);
    int idumi=-1;
    do {
      cpvec(&ps[0],&ps[4]);

      ps[4].x=(ps[0].x+ps[1].x+ps[2].x+ps[3].x)/4.0;
      ps[4].y=(ps[0].y+ps[1].y+ps[2].y+ps[3].y)/4.0;

      dM = sM/sqrt(1.5)*gasdev(&idumi);
      dm = sm/sqrt(1.5)*gasdev(&idumi);
      if (1) {
	ps[4].x += (dM*cos(theta)-dm*sin(theta));
	ps[4].y += (dM*sin(theta)+dm*cos(theta));
      } else {
	ps[4].x += min_rms/sqrt(1.5)*gasdev(&idumi);
	ps[4].y += min_rms/sqrt(1.5)*gasdev(&idumi);
      }

      drift2pix(&ps[4],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,n_vals,&x[4],&y[4]);
      // look at all combinations of 4 positions from the 5 computed
      // to report RMS of the starting position if the 4 starting points
      // occupy each of the 4 pixels we need to bracket.
      {
	int skip,pt,comb_flags,min_skip;
	double sx,sxx,sy,syy,sxy,rms;
	min_rms=1e6;

	for (skip=0;skip<5;skip++) {
	  comb_flags= 0x0000;
	  sxy = sx = sxx = sy = syy = 0.0;
	  for (pt=0;pt<5;pt++) {
	    if (pt != skip) {
	      comb_flags |= (0x0001<<((y[pt]-aim_y[vertex]+1)*2+(x[pt]-aim_x[vertex]+1)));
	      sx += ps[pt].x;		        sy += ps[pt].y;
	      sxx += (ps[pt].x*ps[pt].x);	syy += (ps[pt].y*ps[pt].y);
	      sxy += (ps[pt].x*ps[pt].y);
	    } else {
	      // do nothing, this pixel is being skipped
	    }
	  }
	  if (comb_flags == 0x000f) {
	    sx /= 4;	      sy /= 4;	      sxx /= 4;	      syy /= 4;
	    sxy /= 4;
	    sxx -= pow(sx,2);	              syy -= pow(sy,2);
	    sxy -= sx*sy;
	    theta=0.5*atan2(2*sxy,sxx-syy);
	    // semimajor & semiminor axes:
	    sM = sqrt(0.5*(sxx+syy + 0.3*sqrt(pow(sxx-syy,2)+pow(2*sxy,2))));
	    sm = sqrt(0.5*(sxx+syy - 0.3*sqrt(pow(sxx-syy,2)+pow(2*sxy,2))));

	    rms = sqrt(pow(sM,2)+pow(sm,2));
	    if (rms<min_rms) {
	      min_rms=rms;
	      min_skip=skip;
	    }
	  }
	}
	//	      fprintf(stderr,"min_rms=%g min_skip=%d\n",min_rms,min_skip);
	// repopulate ps[0-3] by skipping over min_skip
	int r_ix=0;
	for (pt=0;pt<5;pt++) {
	  if (pt != min_skip) {
	    // swap ps[pt] & ps[r_ix] and swap {x,y}[pt] & {x,y}[r_ix]
	    // unless want to rerun pixel identification.
	    //		fprintf(stderr,"swap %d & %d\n",pt,r_ix);
	    vec tmp_vec;
	    cpvec(&ps[pt],&tmp_vec);	      
	    cpvec(&ps[r_ix],&ps[pt]);
	    cpvec(&tmp_vec,&ps[r_ix]);
	    // do the same for x coords
	    int tmp_addr;
	    tmp_addr=x[pt];
	    x[pt]=x[r_ix];
	    x[r_ix]=tmp_addr;
	    // and for y coords
	    tmp_addr=y[pt];
	    y[pt]=y[r_ix];
	    y[r_ix]=tmp_addr;
	    r_ix++;
	  }
	}
	// generate a new starting point and store in ps[4]. repeat.
	cpvec(&ps[0],&mean_crn[vertex]);
	mean_crn[vertex].x=(ps[0].x+ps[1].x+ps[2].x+ps[3].x)/4.0;
	mean_crn[vertex].y=(ps[0].y+ps[1].y+ps[2].y+ps[3].y)/4.0;
      }
      if (1) {
	fprintf(stderr,"\rmin_rms=%g sM=%g sm=%g theta=%f\t",min_rms,sM,sm,theta);
      } else {
	fprintf(stderr,"\rmin_rms=%g\t",min_rms);
      }
    } while (min_rms>/* 3e-7 */ 1e-6); // 1e-7 cm = 1nm

    fprintf(stderr,"%s\n",show_vector("mean_corner",&mean_crn[vertex]));
    cpvec(&mean_crn[vertex],&crn->corner[vertex]);
  }

  fprintf(stderr,"from (x,y)=(%d,%d) to (%d,%d)\n",
	  aim_x[0],aim_y[0],
	  aim_x[1],aim_y[1]);
  // now trace. set up a bnd type calculation.
  {
    int pt_ix;
    int betw_pix[]={(crn->way==BND_ROW)?aim_y[0]-1:aim_x[0]-1,
		    (crn->way==BND_ROW)?aim_y[0]  :aim_x[0]};
    
    // fill in the corner values at the head & tail of the array
    crn->dist_along_edge[0]=
      (crn->way==BND_ROW)?mean_crn[0].x:mean_crn[0].y;
    crn->dist_along_edge[crn->n_elem-1]=
      (crn->way==BND_ROW)?mean_crn[1].x:mean_crn[1].y;
    crn->edge_distortion[0]=
      (crn->way==BND_ROW)?mean_crn[0].y:mean_crn[0].x;
    crn->edge_distortion[crn->n_elem-1]=
      (crn->way==BND_ROW)?mean_crn[1].y:mean_crn[1].x;

    // then fill in the rest
    for (pt_ix=1;pt_ix<crn->n_elem-1;pt_ix++) {
      crn->dist_along_edge[pt_ix]=((crn->way==BND_ROW)?mean_crn[0].x:mean_crn[0].y)+
	(pt_ix/((float)crn->n_elem-1)*
	 ((crn->way==BND_ROW)?(mean_crn[1].x-mean_crn[0].x):
	  (mean_crn[1].y-mean_crn[0].y)));
      vec startpos,sp[2],pinchpos;
      int sx[2],sy[2];
      
      startpos.z=ccd_rpp->ccdpars.z[0];

      if (pt_ix==0) {
	startpos.x=
	  (crn->way==BND_ROW)?crn->dist_along_edge[pt_ix]:aim_x[0]*0.01/10.0;
	startpos.y=
	  (crn->way==BND_ROW)?aim_y[0]*0.01/10.0:crn->dist_along_edge[pt_ix];
      } else {
	// take advantage of previous calculation results, careful of any offset
	// applied to dist_along_edge[]
	startpos.x=((crn->way==BND_ROW)?crn->dist_along_edge[pt_ix]:
		    crn->edge_distortion[pt_ix-1]);
	startpos.y=((crn->way==BND_ROW)?crn->edge_distortion[pt_ix-1]:
		    crn->dist_along_edge[pt_ix]);
      }

      cpvec(&startpos,&sp[0]);      cpvec(&startpos,&sp[1]);
      sx[0]=(crn->way==BND_ROW)?0:betw_pix[0];
      sx[1]=(crn->way==BND_ROW)?0:betw_pix[1];
      sy[0]=(crn->way==BND_ROW)?betw_pix[0]:0;
      sy[1]=(crn->way==BND_ROW)?betw_pix[1]:0;;

      target_pix(sx[0],sy[0],&sp[0],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,
		 n_vals,crn->way);
      target_pix(sx[1],sy[1],&sp[1],&ccd_rpp->ccdpars,msi,ccd_rpp->ccdpars.e,
		 n_vals,crn->way);
      if (multipole_stack_pinch(&sp[0],&sp[1],&pinchpos,crn->way,
				&ccd_rpp->ccdpars,ccd_rpp->ccdpars.e,msi) == -1) {
	// didn't converge
      } else {
	// found it.
	crn->edge_distortion[pt_ix]=(crn->way==BND_ROW)?pinchpos.y:pinchpos.x;
	fprintf(stderr,"NOM: (distance,distortion)=(%g,%g)\n",crn->dist_along_edge[pt_ix],crn->edge_distortion[pt_ix]);
      }
    }
  }
}
