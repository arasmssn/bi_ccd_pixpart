#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <argz.h>
#include "henke.h"
#include "ray.h"
#include "numrec/include/nr.h"
#include "numrec/include/nrutil.h"
#include "bi_ccd.h"
#include "xray_bi_ccd.h"

void rebin_map (float *map,int rbn,int *xs,int *ys);
void output_evlist (float *map,int xside,int yside,
		    FILE *of,int *nout,int xoff,int yoff);

void mk_alphas(int nalpha,float alpha[],float bs,float L,float R);
void mk_bs_cumdist(int nrads,float rads[],float maxrad,float cdist[],
		   float z,float bs,float R,float L,int nalpha,float alpha[]);
void dump_local_map(int xs,int ys,float *map);
void get_optical_depths(float *xpos,float t[],float ts[],float imfps[],
			float imfpsks[],float p_si[],float epel[]);
void setup_bsprofs(int nrads,float rads[],float maxrad,int ndists,
		   float *cdists[],float z[],float bs,float R,float L,
		   int nalpha,float alpha[]);
void acis_pix_dist_func(int argz_len,char *argz[],ccd_runpars *ccd_rp);
void acis_pix_dist_complain(void);
void setup_mfps(float energy,float cofda[][4]);
void makeODmaps (int dim);

void xray_bi_ccd (char *xray_arglist,ccd_runpars *ccd_rp) {
  char *argz;
  size_t argz_len;
  int argc;
  char **argv;
  char *entry;
  int i;
  // at the point of this invocation ccd_rp is already 
  fprintf(stderr,"holy smokes: %s\n",xray_arglist);
  if (argz_create_sep(xray_arglist,' ',&argz,&argz_len)) {
    fprintf(stderr,"argz_create failure!\n");
    exit(1);
  }

  argc=argz_count(argz,argz_len);
  argv=(char**)malloc(argc*sizeof(char*));
  entry=NULL;

  for (i=0;i<argc;i++) {
    entry=argz_next(argz,argz_len,entry);
    argv[i]=entry;
  }
  acis_pix_dist_func(argc,argv,ccd_rp);
  return;
}

/* cpix_dist -- ASCA X-Ray CCD event/response simulator by A. Rasmussen */

#define MIN(a,b) (((a)<(b))?(a):(b))

#define PI        3.14159265359
#define PIXEL     13.5     /* pixel size                             */
/* backside event distribution stuff */
#define NDISTS    15    /* number of sample depths */
#define NRADS     300   /* radii   */
#define MAXRAD    300   /* microns */
#define DIFFLEN   400   /* microns */
#define NALPHA    200   /* number of eigenvalues */
#define MINRAD    0.5   /* microns */
#define MINZ      0.000001/* microns */

/* Si characteristics */
#define FANO      0.12     /* fano factor for Si                     */
#define SiYield   0.043    /* fluorescent yield for Si               */
#define ESiK      1.739    /* Si K energy                            */
#define ESiKEdge  1.840    /* Si K shell energy                      */
#define EOKEdge   0.5317   /* O K shell energy                       */
#define ENKEdge   0.4016   /* O K shell energy                       */

/* CCD structure -- updated from ACIS memos 74 & ?? */

#define NA        2.7E12   /* acceptor density                       */
#define WCS       (4.53*10/24.0)   /* width of channel stop covers           */

#define TG1       0.312    /* gate thicknesses                       */
#define TG2       0.266
#define TG3       0.203

#define TGO1      0.315    /* gate oxide thicknesses                 */
#define TGO2      0.256
#define TGO3      0.161

#define G1        (7.800*PIXEL/24.0)  /* gate lengths                           */
#define G2        (7.800*PIXEL/24.0)
#define G3        (7.800*PIXEL/24.0)

#define V31       (0.152*PIXEL/24.0)  /* overlaps                               */
#define V12       (0.150*PIXEL/24.0)
#define V23       (0.125*PIXEL/24.0)

#define S31       (0.20*PIXEL/24.0)   /* gaps between gates                     */
#define S12       (0.20*PIXEL/24.0)
#define S23       (0.20*PIXEL/24.0)

#define TIO       0.06     /* gate insulator oxide thickness         */
#define TIN       0.04     /* gate insulator nitride thickness       */
#define TD        45.0     /* depletion depth                        */
#define TCSP      0.75     /* thickness of channel stop (p++ Si?? )  */
#define TCSO      0.48     /* thickness of channel stop cover (SiO2) */

/* CCD individual characteristics */
/*#define GAIN      3.365*/    /* eV per ADU                             */
/*#define RON       5.5  */    /* read-out noise (electrons)             */
/* old gain values -- originally based on IC443 measurements on all 8 chips. */
/*
float gains[]={
3.565436,3.574005,3.2887644,3.468571,3.1875235,3.2696250,3.517825,3.24945};
*/

/* new gains for s0c1,s0c2, s1c0, s1c3 -- based on W49B (FUJIMOTO). */
/* new gains for other 4 chips are pending W49B data. */

float gains[]={
3.565436,
3.57782, /* Fujimoto - W49B */
3.30033, /* Fujimoto - W49B */
3.468571,
3.19183, /* Fujimoto - W49B */
3.2696250,
3.517825,
3.25098  /* Fujimoto - W49B */
};

/* old noise values -- based on MWB analysis of frame mode data. */
/* float noises[]={5.9,5.5,6.1,6.0,4.4,4.2,5.8,5.0}; */
/* New noise values --- based on GBC measurement of PERSEUS & COMA data */
float noises[]={5.30,5.19,5.41,6.02,4.03,4.12,6.29,4.94}; 

/* miscellaneous */
#define MAXMAP       3000
#define EVENT_THRESH 30

int   nyREG=15;

float yBND[] = {
V31,                  G1-V12,                 G1,                    
G1+S12,               G1+S12+TG2,             G1+S12+TG2+TGO2,  
G1+S12+G2-V23,        G1+S12+G2,              G1+S12+G2+S23,
G1+S12+G2+S23+TG3,    G1+S12+G2+S23+TG3+TGO3, G1+S12+G2+S23+G3-TG3-TGO3,
G1+S12+G2+S23+G3-TG3, G1+S12+G2+S23+G3,       G1+S12+G2+S23+G3+S31};

float SI1[]={
TG1,      TG1,      TG1,      0.0,           TG1+TGO1+TG2, 
TG2,      TG2,      TG2,      0.0,           TG2+TGO2+TG3,
TG3,      TG3,      TG3,      TG3+TG1+TGO1,  0.0};

float SO1[]={
TGO1,           TGO1,     TGO1,           TG1+TGO1,      TGO2,   
TG1+TGO1+TGO2,  TGO2,     TGO1,           TG2+TGO2,      TGO3,    
TG2+TGO2+TGO3,  TGO3,     TG1+TGO1+TGO3,  TGO3,          TG1+TGO1};

float SI2[]={
TG3,    0.0,    TG2,    TG2,    0.0,
0.0,    0.0,    TG3,    TG3,    0.0,
0.0,    0.0,    0.0,    0.0,    TG3};

float SO2[]={
TGO3,   0.0,    TGO2,   TGO2,   0.0,
0.0,    0.0,    TGO3,   TGO3,   0.0,
0.0,    0.0,    0.0,    0.0,    TGO3};

int   ncof=12;
float cofda[][4]={
/* si */            {0.1038,   3.0688,    1.7359,  -1.2153},
/* sio2 */          {0.16533,  3.1152,    1.7406,  -1.1601},
/* si3n4 */         {0.1226,   3.12336,   1.79513, -1.1022},
/* etc. */          {0.2358,   1.22176,  10.3584,   1.6264},
/* below si edge */ {0.91088,  3.02935,   0.4724,  -1.215895},
		    {0.3380,   3.5212,    2.3309,  -1.1949},
/* etc. */          {0.2358,   1.22176,  10.3584,   1.6264},
/* below ok edge */ {0.698319, 2.24294,   4.541946, 2.5228e-5},
		    {0.3380,   3.5212,    2.3309,  -1.1949},
/* etc. */          {0.2358,   1.22176,  10.3584,   1.6264},
/* below nk edge */ {0.698319, 2.24294,  4.541946,  2.5228e-5},
		    {3.5621,   2.30224, -0.1332,   -4.9e-5}};

struct data_str
  {
  char mode;
  int framenum;
  short chipnum,x,y,data[9];
  } event;

float threshold;
short size = sizeof(struct data_str);
float imfp_si,imfp_sio2,imfp_si3n4;       /* imfps for Si, SiO2, Si3N4      */
float imfpsk_si,imfpsk_sio2,imfpsk_si3n4; /* same but for Si K fluorescence */
float si1,si2,so1,so2; 
FILE  *fp,*outf;

/* modes */
enum MODES {PUTOUT_MAPS,PUTOUT_RV,PUTOUT_OPTICALDEPTHS};

void acis_pix_dist_func (int argc,char *argv[],ccd_runpars *ccd_rp)
{
  float *bigmap=NULL;      int NX=2048;      int NY=512;
  int ACCUMULATE_N_FLUSH=0;
  float N_resonance_collisions=0.0;
  float x[2],y[2],z[2],nE[2],nrg[2],evpel[2],nelectrons[3]; 
  /*simulated photon positions wrt pixel center*/
  float dfe,eMobility,lamda,zeta,sigma,init_sigma,drift_sigma,depletiondepth;
  float odval,odval2,odstep,si_yield,minDelta,max_drift_sigma;
  float tau[7],tausk[7],likelihood_Si[7],en_elec[7];
  //  float siktau;
  float cswalldepth[4],ckwalldepth[4],fraction[4],ft[5],fraction_moment[4];
  float imfps[7],imfpsks[7],distance,R;
  float x0,y0,primaryenergy,gain,ron,echo,eloss,elost,tx,ty,typ;
  //  float tz;
  float xc,yc,zc,phi,map[MAXMAP],fracx[128],fracy[128],atten_coeff();
  //  float ta,tb,chi,chi0,ftemp1,ftemp2,ftemp3,ftemp4;
  int temp1,temp2;
  float this,last,usex,usey;
  float temperature,t_bs_Si,ckdeff,deflection(),wallthicknss();
  float P,fdff[2],zm,nkt_kt,diffLen,z_twiddle;
  static float csdefault[][4]={{29.0,3.5,9.0,17.0},{29.0,3.5,9.0,19.0}};
  static float ckdefault[][4]={{20.0,5.0,9.5,18.0},{21.25,4.5,8.25,20.0}};
  int   i,j,k,part,m,n,N,numphot,xside,yside,startx,starty;
  int   idumi,nout,escape,fluoresce;
  int   sensor,chip,dimension;
  long  iduml,ntries,nattempt,nreal,nescape,nfluor;
  char  BSI,cswallspec,ckwallspec,exclude_SiK,exclude_Cont;
  char  fullydepleted;
  enum  MODES operation=PUTOUT_RV;
  float zlim[2],min_delta_scale,drift_sigma_scale;
  float *cdists[NDISTS],alpha[NALPHA],rads[NRADS],zlist[NDISTS];
  int rbn=1;
  int finished_reading=0;
  int do_rays=0;
  ray aray;


  if (argc<3) acis_pix_dist_complain();


  /* defaults */

  sensor=1;  chip=3;
  gain=gains[4*sensor+chip];
  gain=3.65;
  ron =noises[4*sensor+chip];
  ron=2.50;
  sensor=-1; chip=-1;
  depletiondepth=TD;
  dfe=0.0;
  temperature=223.0;
  t_bs_Si=120.0;
  si_yield=SiYield;
  BSI=cswallspec=ckwallspec=fullydepleted=0;
  zlim[0]=zlim[1]=0.0;

  /* these wall depths (and widths) were found to minimize chisquared
     when comparing branching ratios at sp=40ADU, 4.2 electrons readout noise
     for sensor 1 chip 1 -- ISAS data. AR, 27-Jul-93 */

  cswalldepth[0]=0.0;
  cswalldepth[1]=0.0;
  cswalldepth[2]=0.0;
  cswalldepth[3]=0.0;
  ckwalldepth[0]=0.0;
  ckwalldepth[1]=0.0;
  ckwalldepth[2]=0.0;
  ckwalldepth[3]=0.0;
  fdff[0]=0.0;
  fdff[1]=0.0;
  exclude_SiK=0;
  exclude_Cont=0;
  ckdeff=1.0;
  eloss=0.0;
  dimension=128;
  primaryenergy=0.0;
  threshold=(float)(EVENT_THRESH);
  diffLen=DIFFLEN;
  min_delta_scale=1.0;
  drift_sigma_scale=1.0;
  echo=0.0;
  n=0L;
  N=0L;
  P=6.5; /* channel potential, needed for fully depleted backside devices */
  nkt_kt=1.0; /* number of kT to use as the schottky barrier `fieldfree' zone*/
  R=0.00001;
  outf=stdout;

  // here transfer parameters from ccd_runpars to old commandline switch values:
  temperature   =ccd_rp->ccdpars.T;
  depletiondepth = ccd_rp->ccdpars.t_si * 1e4; // because t_si is in cm

  int nsigma=ccd_rp->ccdpars.n_sigma;
  float *lateral_sigma=ccd_rp->ccdpars.lateral_sigma;


  while (argc) {
    if ((argv[0][0]=='-') && 
	(((strchr("BiCFRx",argv[0][1])!=NULL) && (argc>=1)) ||
	 ((strchr("z",argv[0][1])!=NULL) && (argc>=3)) ||
	 ((strchr("w",argv[0][1])!=NULL) && (argc>=5)) ||
	 ((strchr("ADLMNPRSYbcefgklmnorst",argv[0][1])!=NULL) && (argc>=2)))) {

      switch (argv[0][1]){
	/* attempt to specify diagnostic output. */
      case 'i':
	argc--;argv++;
	do_rays = 1; 
	break; /* expect rays on stdin */
      case 'B':
	argc--;argv++;
	BSI = 1; 
	break; /* backside illuminated*/
      case 'C': 
	argc--;argv++;
	exclude_Cont=1;
	break; /* no partial events */
      case 'D': 
	argc--;argv++;
	dfe      =atof(argv[1]);
	break;
      case 'F': 
	argc--;argv++;
	fullydepleted=1;
	break;/*device is fully dep'd*/
      case 'A':
	argc--;argv++;
	ACCUMULATE_N_FLUSH=atoi(argv[0]);
	argc--;argv++;
	break; /* accumulate x-rays onto the frame. */
      case 'L': 
	argc--;argv++;
	diffLen=atof(argv[0]);
	argc--;argv++;
	break;
      case 'M':
	argc--;argv++;
	min_delta_scale=atof(argv[0]);
	argc--;argv++;break;
      case 'N': 
	argc--;argv++;
	N=atol(argv[0]);
	argc--;argv++;
	break;
      case 'P': 
	argc--;argv++;
	P=atof(argv[0]);
	argc--;argv++;
	break;/* weighted channel potential, volts */
      case 'R': 
	argc--;argv++;
	N_resonance_collisions=atof(argv[0]);
	argc--;argv++;
	break;
	//      case 'T':
	//	argc--;argv++;
	//	temperature=atof(argv[0]); 
	//	argc--;argv++;
	//	break;
      case 'S': 
	argc--;argv++;
	drift_sigma_scale=atof(argv[0]); 
	argc--;argv++;
	break;
      case 'Y':
	argc--;argv++;
	si_yield =atof(argv[0]);     
	argc--;argv++;
	break;
      case 'b': 
	switch (argv[0][2]) {
	case 'r':
	  /* backside reflection coefficient */
	  argc--;argv++;
	  R=atof(argv[0]);
	  argc--;argv++;
	  if (R<=0.0 || R>=1.0) {
	    fprintf(stderr,"reflection coefficient must be 0<R<1.\n");
	    acis_pix_dist_complain();
	  }
	  break;
	case 'n':
	  // specify rebinning
	  argc--;argv++;
	  rbn=atoi(argv[0]);
	  argc--;argv++;
	  break;
	default:
	  argc--;argv++;
	  t_bs_Si  =atof(argv[0]);
	  argc--;argv++;
	  if (t_bs_Si<=0.0) {
	    fullydepleted=1;
	    t_bs_Si=0.0;
	  }
	  break;
	}
	break; /* backside undepleted */
      case 'c':
	argc--;argv++;
	chip=atoi(argv[0]);
	argc--;argv++;
	break;
	//      case 'd':
	//	argc--;argv++;
	//	depletiondepth=atof(argv[0]);
	//	argc--;argv++;
	//	break;
      case 'e':
	argc--;argv++;
	primaryenergy=atof(argv[0]);
	argc--;argv++;
	break;
      case 'f':
	argc--;argv++;
	ckdeff   =atof(argv[0]);
	argc--;argv++;
	break;
	/* attempt to specify gain, eV per ADU */
      case 'g': 
	argc--;argv++;
	gain     =atof(argv[0]);     
	argc--;argv++;
	break;
      case 'k': 
	switch (argv[0][2]) {
	case 'T':   
	  argc--;argv++;
	  nkt_kt=atof(argv[0]);  
	  argc--;argv++;
	  break;
	default:	  
	  argc--;argv++;
	  echo  =atof(argv[0]);
  	  argc--;argv++;
	  break;
	}
	break;
      case 'l':
	argc--;argv++;
	eloss    =atof(argv[0]);
	argc--;argv++;
	break;
      case 'm': 
	argc--;argv++;
	{
	  switch (argv[0][0]){
	  case 'o':
	    if (argc>=2) {
	      operation=PUTOUT_OPTICALDEPTHS;
	      argc--;argv++;
	      if (argv[0][0]!='-') {
		dimension=atoi(argv[0]);
	      }
	    } else {
	      fprintf(stderr,"wrong number or args??");
	      acis_pix_dist_complain();
	    }
	    break;
	  case 'e':
	    operation=PUTOUT_MAPS;
	    break;
	  case 'r':
	    operation=PUTOUT_RV;
	    break;
	  default:
	    acis_pix_dist_complain();
	    break;
	  }
	  argc--;argv++;
	}
	break;
      case 'n': 
	if (argv[0][2]=='e') {
	  argc--;argv++;
	  ron=atof(argv[0]);
	  argc--;argv++;
	} else {
	  argc--;argv++;
	  n=atol(argv[0]);
	  argc--;argv++;
	}
	break;
      case 'o': 
	argc--;argv++;
	if ((outf=fopen(argv[0],"w")) == NULL) 
	  acis_pix_dist_complain(); 
	argc--;argv++;
	break;
      case 'r':
	argc--;argv++;
	ron=atof(argv[0]);
	argc--;argv++;
	break;    
      case 's': 
	argc--;argv++;
	sensor=atoi(argv[0]);
	argc--;argv++;
	break;
      case 't': 
	argc--;argv++;
	threshold=atof(argv[0]);     
	argc--;argv++;
	break;
	/* attempt to specify wall depths */
      case 'w': 
	switch (argv[0][2]) {
	case 'c':
	  argc--;argv++;
	  cswalldepth[0]=atof(argv[0]);
	  argc--;argv++;
	  cswalldepth[1]=atof(argv[0]);
	  argc--;argv++;
	  cswalldepth[2]=atof(argv[0]);
	  argc--;argv++;
	  cswalldepth[3]=atof(argv[0]);
	  argc--;argv++;
	  cswallspec=1;
	  break;
	case 'k':
	  argc--;argv++;
	  ckwalldepth[0]=atof(argv[0]);
	  argc--;argv++;
	  ckwalldepth[1]=atof(argv[0]);
	  argc--;argv++;
	  ckwalldepth[2]=atof(argv[0]);
	  argc--;argv++;
	  ckwalldepth[3]=atof(argv[0]);
	  argc--;argv++;
	  ckwallspec=1;
	  break;
	default:
	  argc--;argv++;
	  fprintf(stderr,"Unknown: %s\n",argv[0]);
	  acis_pix_dist_complain();
	  break;
	}
	break;
      case 'x': 
	exclude_SiK=1;
	argc++;argv--; 
	break; /* no fluor & escape */
      case 'z': 
	argc++;argv--; 
	zlim[0]=atof(argv[0]);
	argc++;argv--; 
	zlim[1]=atof(argv[0]);
	argc--;argv++;
	break;
      default : 
	acis_pix_dist_complain();
	break;
      }
    } else {
      acis_pix_dist_complain();
    }
  }

  eMobility=1450.0*pow(temperature/300.0,-2.42); 
  minDelta=min_delta_scale*sqrt(5.58*temperature*2e12/(200.0*NA));
  //  max_drift_sigma=3.34117*drift_sigma_scale*
  //    sqrt(-log(1.-(minDelta/depletiondepth))*(temperature/200.0)*(2e12/NA));
  max_drift_sigma=lateral_sigma[0];

  switch (sensor) {
  case 0:
  case 1:
    for (j=0;j<4;j++) {
      if (! ckwallspec ) ckwalldepth[j]=ckdefault[sensor][j];
      if (! cswallspec ) cswalldepth[j]=csdefault[sensor][j];
    }
    break;
  default:
    break;
  }


  if (fullydepleted && !BSI) {
    acis_pix_dist_complain();
  }
  if (fullydepleted && (P==0.0)) {
    fprintf(stderr,"specify channel potential for fullydepleted devices.\n");
    acis_pix_dist_complain();
  }
  if (fullydepleted) {
    t_bs_Si=0.0;
  }

  if (zlim[0]==0.0 && zlim[1]==0.0) {
    if (BSI) {
      if (fullydepleted) {
	float k,q,e0,esi,efermi,Ebs,xm,A,zeta1,zeta2;
	/* here calculate the schottky barrier parameters,
	   together with temperature, to formulate a field-free
	   region as well as a `dead' layer (where charge is not collected) */
	/* use: PtSi -- p-type Si schottky barrier of 0.2 volts */
	/* xm=sqrt(q/(16*Pi*e0*esi*E(bs))) */
	q=1.6e-19;
	k=1.38e-16;
	e0=8.86e-14;
	esi=11.7;
	efermi=0.20;
	Ebs=(P+efermi+0.5*(q*NA)/(esi*e0)*pow(1e-4*depletiondepth,2.0))/
	  (1e-4*depletiondepth); /*volts/cm*/
	/* z_twiddle is the projected location (measured from the back surface)
	   where the electric field should drop to zero. (effective depletion
	   depth for charge diffusion purposes */
	z_twiddle=1e4*((P+efermi)*(esi*e0)/(1e-4*depletiondepth*q*NA)
		       -0.5*1e-4*depletiondepth);          /*microns*/
	xm=sqrt(q/(16*PI*e0*esi*Ebs)); /* cm */
	A = (nkt_kt*k*temperature*1.0e-7)/(2*q*Ebs*xm); 
	zeta1=1+A-sqrt(pow(1+A,2.0)-1);
	zeta2=1+A+sqrt(pow(1+A,2.0)-1);
	/* thickness of the `dead' layer */
	fdff[0] = zeta1*xm*1.0e4; /* microns */
	/* boundary of the `fieldfree' region (typ. 0.1 microns) */
	fdff[1] = zeta2*xm*1.0e4; /* microns */
	zlim[0]=0.0;
	zlim[1]=depletiondepth;
	zm=xm*1.0e4;
      } else { /* not fully depleted */
	zlim[0]=-t_bs_Si;
	zlim[1]=depletiondepth;
	z_twiddle=0.0;
      }
    } else {
      zlim[0]=-5.0;
      zlim[1]=depletiondepth+t_bs_Si;
      z_twiddle=0.0;
    }
  } else if (zlim[0]<-5.0||zlim[0]>zlim[1]||zlim[1]>depletiondepth+t_bs_Si){
    fprintf(stderr,"zlim must be ordered and within depletion region.");
    acis_pix_dist_complain();
  }

  if (!do_rays && (primaryenergy==0.0)) acis_pix_dist_complain();
  if (operation == PUTOUT_OPTICALDEPTHS) {
    setup_mfps(primaryenergy,cofda);
    makeODmaps(dimension);exit(0);
  }
  if (!do_rays && (primaryenergy==0.0)) acis_pix_dist_complain();
  if (!do_rays && ((n==0L && N==0L) || ((n & N) != 0L)))
    acis_pix_dist_complain(); 
  if ((sensor==-1) ^ (chip==-1)) acis_pix_dist_complain();
  if (sensor!=-1) {
    gain=gains [4*sensor+chip];
    ron =noises[4*sensor+chip];
  }

/*   energy        = primaryenergy; */
  ntries        = nreal = nattempt = nescape = nfluor = 0L;
  
/* regurgitate input parameters */
  fprintf(stderr,"gain = %f\n",gain);
  fprintf(stderr,"energy= %f\n",primaryenergy);
  fprintf(stderr,"number=%d\n",n);
  fprintf(stderr,"sensor=%d\n",sensor);
  fprintf(stderr,"chip=%d\n",chip);
  fprintf(stderr,"noise=%f\n",ron);
  fprintf(stderr,"electron loss=%f\n",eloss);
  fprintf(stderr,"depletiondepth=%f\n",depletiondepth);
  fprintf(stderr,"event threshold=%f\n",threshold);
  fprintf(stderr,"echo=%f\n",echo);
  fprintf(stderr,"CS wall width=%f d1=%f d2=%f d3=%f\n",
	  cswalldepth[0],cswalldepth[1],cswalldepth[2],cswalldepth[3]);
  fprintf(stderr,"CLK wall width=%f d1=%f d2=%f d3=%f\n",
	  ckwalldepth[0],ckwalldepth[1],ckwalldepth[2],ckwalldepth[3]);
  fprintf(stderr,"zlims=%f,%f\n",zlim[0],zlim[1]);
  fprintf(stderr,"temperature = %f\n",temperature);
  fprintf(stderr,"dark frame error = %f electrons.\n",dfe);
  fprintf(stderr,"backside undepleted region = %f microns.\n",t_bs_Si);
  fprintf(stderr,"backside reflection coefficient = %f\n",R);
  fprintf(stderr,"backside diffusion length = %f\n",diffLen);
  fprintf(stderr,"backside illuminated? %d.\n",BSI);
  fprintf(stderr,"exclude_siK? %d\n",exclude_SiK);
  fprintf(stderr,"exclude_continuum? %d\n",exclude_Cont);
  fprintf(stderr,"minimum delta: %f\n",minDelta);
  fprintf(stderr,"Si_Yield=%f\n",si_yield);
  if (BSI) {
    fprintf(stderr,"fully depleted? %d\n",fullydepleted);
    fprintf(stderr,"energy threshold: %f * kT.\n",nkt_kt);
    fprintf(stderr,"effective `dead' layer: 0.0 - %f microns.\n",fdff[0]);
    fprintf(stderr,"`field-free' region : %f - %f microns.\n",fdff[0],fdff[1]);
    fprintf(stderr,"weighted channel potential=%f\n",P);
    fprintf(stderr,"z_twiddle=%f\n",z_twiddle);
  }

  setup_mfps(primaryenergy,cofda);


  for(i=0;i<NDISTS;i++) 
    if ((cdists[i]=(float*)malloc(NRADS*sizeof(float)))==NULL)
      exit(1);

  if (fullydepleted) {
    setup_bsprofs((int)(NRADS),rads,(float)(MAXRAD),(int)(NDISTS),
		  cdists,zlist,fdff[1]-fdff[0],(BSI)?R:0.00001,diffLen,
		  (int)(NALPHA),alpha);
  } else {
    setup_bsprofs((int)(NRADS),rads,(float)(MAXRAD),(int)(NDISTS),
		  cdists,zlist,t_bs_Si,(BSI)?R:0.00001,diffLen,
		  (int)(NALPHA),alpha);
  }

  /* initialize random sequence */
  time(&iduml);
  ran2(&iduml);
  idumi=-1;
  expdev(&idumi);
  gasdev(&idumi);

  /* do the optical depth calculation for BSI only once. */
  if (BSI) {
    tau[0]=t_bs_Si*imfp_si;tausk[0]=t_bs_Si*imfpsk_si;
    /* fudge optical depths calculation because of backside illumination */
    for(k=1;k<7;k++){tau[k]=0.0;tausk[k]=0.0;}
  }



  i=0;
  finished_reading=0;

  fprintf(stderr,"do_rays is %d\n",do_rays);

  while ( (!do_rays && (i < n || ntries < N)) || do_rays ) {
    if (!do_rays) {
      ntries++;
      /* reset energy value */
      nrg[0] = primaryenergy; 
      nrg[1] = 0.0;
      numphot=1;escape=0;fluoresce=0;
      /* sample the spatial distribution.. */
      x[0]= PIXEL * (-0.5 + ran2(&iduml));
      y[0]= PIXEL * (-0.5 + ran2(&iduml));
      
      /* a more convenient representation for LE QE stuff. */
      y0 = PIXEL/2. + y[0];
      x0 = PIXEL - fabs((double)(2.*x[0]));

      y0 -= PIXEL*floor(y0/PIXEL);
      x0 -= PIXEL*floor(x0/PIXEL);

    } else {
      // do_rays. read in from stdin.
      if (fread(&aray,sizeof(ray),1,stdin)) {
	// collapse the ray on the surface and proceed to 
	// determine its interaction position
	float kev=1.2398*modulus(&aray.k)/(0.1*2*M_PI);
	primaryenergy=kev;
	nrg[0] = kev;
	nrg[1] = 0.0;
	numphot=1;escape=0;fluoresce=0;
	x[0]=aray.p.x/1e-3; // in microns
	y[0]=aray.p.y/1e-3; // in microns

	y0 = PIXEL/2. + y[0];
	x0 = PIXEL - fabs((double)(2.*x[0]));
	y0 -= PIXEL*floor(y0/PIXEL);
	x0 -= PIXEL*floor(x0/PIXEL);

	setup_mfps(nrg[0],cofda);
      } else {
	finished_reading=1;
	// going straight to finished_rdng (e.g., EOF) causes a core dump
	goto finished_rdng; 
      }
    }

    /* locate the y value of the event */

    j=0;
    while ( j<nyREG && y0 > yBND[j] ) j++;
    si1=SI1[j]; 
    so1=SO1[j];
    si2=SI2[j];
    so2=SO2[j];

    /* optical depth calculation */
    if (! BSI) {
      get_optical_depths(&x0,tau,tausk,imfps,imfpsks,likelihood_Si,en_elec);
      j=6;
    } else {
      //      imfps[0]=imfp_si;
      //      en_elec[0]=0.00365;
      // no window material or anything. backside illuminated
      j=-1;
    }

    odval=expdev(&idumi);

    float cosine;

    if (do_rays) {
      vec tmp,norm;
      norm.x=norm.y=0.0;
      norm.z=1;
      memcpy(&tmp,&aray.k,sizeof(vec));
      cosine=fabs(dot_prod(&tmp,&norm)/modulus(&tmp));
      fprintf(stderr,"REM cosine %g\n",cosine);
    } else {
      cosine=1.0;
    }
    //    fprintf(stderr,"r[0] %f cosine %f\n",sqrt(pow(x[0],2)+pow(y[0],2)),cosine);
    /* see if photon made it through the gate structure. If not, there is
       a possibility for the fluorescence to be detected.. */
    while (j>=0 && odval>tau[j]/cosine) {
      // this is computed in the case of FSI
      odval = odval-tau[j]/cosine;j--;
      // neglect the change in position as photon traverses
      // through gate structure
    }

    if (j<0) { 
      /* photon gets into depleted region */
      z[0]= odval/imfp_si * cosine;
      fprintf(stderr,"REM depth is z[0]=%g\n",z[0]);
      if (do_rays) {
	x[0] += odval/imfp_si * (aray.k.x)/modulus(&aray.k);
	y[0] += odval/imfp_si * (aray.k.y)/modulus(&aray.k);
      }
      fprintf(stderr,"REM %f %f\n",sqrt(pow(x[0],2)+pow(y[0],2)),z[0]);
      /* z[m] is measured from the depletion region boundary on the side */
      /* on which the photon entered.  */

      if (z[0]>0.0 && z[0]<depletiondepth)
	i++;                           /* increment # `interacting' photons */

      fprintf(stderr,"REM compare z[0] to zlim[1]=%g and zlim[0]=%g\n",zlim[1],zlim[0]);

      if ( (z[0] > zlim[1]) || (z[0] <= zlim[0])) continue;

      /* the photon interacted in depleted region !! continue.. */

      evpel[0]=0.00365;

      /* consider escape possibility. */

      if ((primaryenergy > ESiKEdge) && 
	  (!exclude_SiK) && 
	  (ran2(&iduml) < si_yield)) {
	numphot=2;
	escape=1;
	nrg[0] = primaryenergy - ESiK;
	nrg[1] = ESiK;
	zc = 1. - 2. * ran2(&iduml);
	odval2=expdev(&idumi);
	z[1] = z[0] + zc*odval2/imfpsk_si;
	phi=6.28319*ran2(&iduml);
	xc = sqrt(1. - zc*zc) * cos(phi);
	yc = xc * tan(phi);
	x[1] = x[0] + odval2/imfpsk_si * xc;
	y[1] = y[0] + odval2/imfpsk_si * yc;
	evpel[1]=0.00365;
	/* this above wasn't treated completely. We are neglecting 
	   material properties of the gate structure, where fluorescent
	   photons may end up. i.e. assuming that the MFP is that appropriate
	   for silicon, and the energy per electron produced is the straight 
	   0.00365 keV. */ 
      }
    } else { 
      /* the photon didn't get through. */
      /* how far from the boundary did it land? */
      distance=(tau[j]-odval)/imfps[j];
      for (k=j;k>=0;k--) {
	if (k==j) continue;
	distance+=tau[k]/imfps[k];
      }
      z[0] = -distance; /* this far from the boundary. */
      if ( z[0] > zlim[1] || z[0] <= zlim[0]) continue;
      evpel[0]=en_elec[j];
      /* chances SILICON fluorescent photon */
      if ((primaryenergy > ESiKEdge) && 
	  (ran2(&iduml) < likelihood_Si[j]*si_yield )) {
	if (exclude_SiK) continue;
	numphot=2;
	nrg[0] = primaryenergy - ESiK;
	nrg[1] = ESiK;
	x[1]=x[0];y[1]=y[0];z[1]=z[0]; /* initial values */
	/* see if the fluorescent photon makes it into depletion region. */
	zc = 1. - 2*ran2(&iduml); /* photon may be going either direction */
	phi=6.28319*ran2(&iduml);
	xc = sqrt(1. - zc*zc) * cos(phi);
	yc = xc * tan(phi);
	/* new optical depth calculation */
	odval2=expdev(&idumi);
	if (zc<0.0) odstep=MIN((tau[j]-odval)*tausk[j]/tau[j],odval2);
	else odstep=MIN(odval*tausk[j]/tau[j],odval2);
	z[1] += zc*odstep/imfpsks[j];
	x[1] += xc*odstep/imfpsks[j];
	y[1] += yc*odstep/imfpsks[j];
	odval2-=odstep;
	evpel[1]=en_elec[j];
	for (k=j; !BSI && k<7 && k>=0 && odval2>0.0 ;k += (zc<0.0)?1:-1) {
	  if (k==j) continue;
	  odstep=MIN(tausk[k],odval2);
	  z[1] += zc*odstep/imfpsks[j];
	  x[1] += xc*odstep/imfpsks[j];
	  y[1] += yc*odstep/imfpsks[j];
	  odval2-=odstep;
	  evpel[1]=en_elec[k];
	}
	/* if photon has enough mmpf it may end up in the depletion region */
	if (odval2>0.0) {
	  z[1]+=zc*odval2/imfpsk_si;
	  x[1]+=xc*odval2/imfpsk_si;
	  y[1]+=yc*odval2/imfpsk_si;
	  evpel[1]=0.00365;
	  /* check on interaction positions later */
	}
	/* now have the interaction position of fluorescence photon */
      } 
    } 
    

 
//diagnostic output
//printf("numphot=%d ",numphot);
//    for (m=0;m<numphot;m++) 
//      printf("%f %f %f %f ",x[m],y[m],z[m],nrg[m]);
//    printf("\n");
//continue; 


    /* final primary energy..  calculate # of electrons produced. */
    for(m=0;m<numphot;m++) {
      /* use nominal notation */
      if (BSI) z[m]=depletiondepth-z[m]; 
      /* electron number calculation */
      nE[m]=(int)((nrg[m] / 0.00365) + 
		  sqrt(FANO * nrg[m] / evpel[m]) * gasdev(&idumi) + 0.5);
    }
    /* time to re-group. how many events will be detected? */
    /* first calculate the number of pixels spanned between event locations */

    switch (numphot) {
    case 1:
      temp1 = (int)(x[0]/PIXEL+0.5);
      temp2 = (int)(y[0]/PIXEL+0.5);
      startx = temp1 - 2*rbn;
      starty = temp2 - 2*rbn;
      //      startx=(floor(temp1/rbn)-2)*rbn;
      //      starty=(floor(temp2/rbn)-2)*rbn;
      xside=yside=5*rbn;
      break;
    case 2:
      temp1 = (int)(x[0]/PIXEL+0.5);
      temp2 = (int)(x[1]/PIXEL+0.5);
      startx= ((temp1<temp2)?temp1:temp2)-2*rbn;
      xside = abs(temp1-temp2)+5;
      xside *=rbn;
      temp1 = (int)(y[0]/PIXEL+0.5);
      temp2 = (int)(y[1]/PIXEL+0.5);
      starty= ((temp1<temp2)?temp1:temp2)-2*rbn;
      yside = abs(temp1-temp2)+5;
      yside *=rbn;
      if (xside*yside>=MAXMAP) {
	startx=starty=-2;
	xside=yside=5;
	startx=starty=-2*rbn;
	xside=yside=5*rbn;
      }
      break;
    }
    /* initialize charge maps */
    fprintf(stderr,"REM MAXMAP=%d vs. xside*yside=%d\n",MAXMAP,xside*yside);
    for (j=0;j<xside*yside;j++) map[j]=0.0;

    /* first collect the true number of electrons using the map. */
    for (m=0;m<numphot;m++) {

      /* fudge for now, since field free region drift is not understood */
      /*if ((z[m]>zlim[1]) || (z[m]<=zlim[0])) continue;*/
      if (z[m]>depletiondepth+t_bs_Si) continue;  /* this prevents crashing */
      usex=x[m];usey=y[m];
      /* idea here is to shift charge cloud around rather than use different
	 sigma X and sigma Y expressions. */
      /* include initial charge cloud. */
      /* ready to go on.. */

      /* assume for now that the charge cloud size may be calculated from
	 the amount of energy deposited (e.g. from escape) */

      init_sigma=(1.03*1.7e-2*pow(nE[m]*0.00365,1.75))/2.0; /* hopkinson 2,3 */
      /* modification inspired by scholtz&ulm */
      init_sigma=sqrt(N_resonance_collisions*0.1*0.1+init_sigma*init_sigma);
      /* What is the fraction of charge contained in the depletion region? */
      /* want to make this so that all charge at bottom is collected ?? */

      if (fullydepleted && BSI) {
	fprintf(stderr,"REM fullydepleted && BSI.\n");
	/* note : t_bs_Si is out of the picture here */
	fraction[1]=erf(z[m]/(sqrt(2.0)*init_sigma));
	fraction[2]=erf((z[m]-(depletiondepth-fdff[1]))/
			(sqrt(2.0)*init_sigma));
	fraction[3]=erf((z[m]-(depletiondepth-fdff[0]))/
			(sqrt(2.0)*init_sigma));
	/* fraction of charge that is `missed' - outside the depleted zone */
	fraction[0]=0.5*(1.0-fraction[1]);
	/* fraction of charge that will be collected in drift zone */
	fraction[1]=0.5*(fraction[1]-fraction[2]);
	/* fraction of charge that 
	   must diffuse between 2 absorbing boudaries */
	fraction[2]=0.5*(fraction[2]-fraction[3]);
	/* fraction of charge that is gone forever sucked by the schottky 
	   barrier/image field */
	fraction[3]=0.5*(fraction[3]+1.0);
	/* moments of the charge within each zone */
	ft[0]=0.0;
	ft[1]=exp(-(z[m]*z[m])/(2.0*init_sigma*init_sigma));
	ft[2]=exp(-((z[m]-(depletiondepth-fdff[1]))*
		    (z[m]-(depletiondepth-fdff[1])))/
		  (2.0*init_sigma*init_sigma));
	ft[3]=exp(-((z[m]-(depletiondepth-fdff[0]))*
		    (z[m]-(depletiondepth-fdff[0])))/
		  (2.0*init_sigma*init_sigma));
	ft[4]=0.0;
	fprintf(stderr,"REM ok, got here.\n");
      } else {

	/* usual treatment fror frontside & backside with finite field-free
	   zones */
	if ( ! exclude_Cont ) {
	  fraction[1]=erf(z[m]/(sqrt(2.0)*init_sigma));
	  fraction[2]=erf((z[m]-depletiondepth)/(sqrt(2.0)*init_sigma));
	  fraction[3]=erf((z[m]-depletiondepth-t_bs_Si)/
			  (sqrt(2.0)*init_sigma));
	  fraction[0]=0.5*(1.0-fraction[1]);
	  fraction[1]=0.5*(fraction[1]-fraction[2]);
	  fraction[2]=0.5*(fraction[2]-fraction[3]);
	  fraction[3]=0.5*(fraction[3]+1.0);
	  ft[0]=0.0;
	  ft[1]=
	    exp(-(z[m]*z[m])/(2.0*init_sigma*init_sigma));
	  ft[2]=exp(-((z[m]-depletiondepth)*(z[m]-depletiondepth))/
		    (2.0*init_sigma*init_sigma));
	  ft[3]=exp(-((z[m]-depletiondepth-t_bs_Si)*
		      (z[m]-depletiondepth-t_bs_Si))/
		    (2.0*init_sigma*init_sigma));
	  ft[4]=0.0;
	} else {
	  fraction[1]=erf(z[m]/(sqrt(2.0)*0.0001));
	  fraction[2]=erf((z[m]-depletiondepth)/(sqrt(2.0)*0.0001));
	  fraction[3]=erf((z[m]-depletiondepth-t_bs_Si)/(sqrt(2.0)*0.0001));
	  fraction[0]=0.5*(1.0-fraction[1]);
	  fraction[1]=0.5*(fraction[1]-fraction[2]);
	  fraction[2]=0.5*(fraction[2]-fraction[3]);
	  fraction[3]=0.5*(fraction[3]+1.0);
	  ft[0]=0.0;
	  ft[1]=
	    exp(-(z[m]*z[m])/(2.0*0.0001*0.0001));
	  ft[2]=exp(-((z[m]-depletiondepth)*(z[m]-depletiondepth))/
		    (2.0*0.0001*0.0001));
	  ft[3]=exp(-((z[m]-depletiondepth-t_bs_Si)*
		      (z[m]-depletiondepth-t_bs_Si))/
		    (2.0*0.0001*0.0001));
	  ft[4]=0.0;
	}
      }
      
      fprintf(stderr,"REM wahoo!\n");

      {
	int jbl;
	for (jbl=0;jbl<4;jbl++) {
	  if (fraction[jbl] == 0.0) {
	    /* can't do the calculation */
	    switch (jbl) {
	    case 1:
	      fraction_moment[jbl]=0.5*depletiondepth;
	      break;
	    case 2:
	      fraction_moment[jbl]=(fullydepleted)?
		depletiondepth-0.5*(fdff[0]+fdff[1]):
		  depletiondepth+0.5*t_bs_Si;
	      break;
	    default:
	      fraction_moment[jbl]=z[m];
	      break;
	    }
	  } else {

/* there might be some ambiguity here - if fullydepleted, exclude_cont
   shouldn't be available, in principle. there's no safeguard at present. */

	    switch (exclude_Cont) {
	    case 1:
	      fraction_moment[jbl]=
		z[m]+0.0001*(sqrt(2.0/PI))*(ft[jbl]-ft[jbl+1])/fraction[jbl];
	      break;
	    default:
	      fraction_moment[jbl]=
	z[m]+init_sigma*(sqrt(2.0/PI))*(ft[jbl]-ft[jbl+1])/fraction[jbl];
	      break;
	    }
	    switch (jbl) {
	    case 1:
	      if (fraction_moment[jbl]>depletiondepth-minDelta)
		fraction_moment[jbl]=depletiondepth-minDelta;
	      break;
	    case 2:
/*	      if (fraction_moment[jbl]<depletiondepth+minDelta)
		fraction_moment[jbl]=depletiondepth+minDelta;
	      if (fraction_moment[jbl]>depletiondepth+t_bs_Si)
		fraction_moment[jbl]=depletiondepth+t_bs_Si;
*/	      break;
	    default:
	      break;
	    }
	  }    
	}
      }

      fprintf(stderr,"REM wahoo 2!\n");


      /* nelectrons[0] is just a placeholder, never used */
      nelectrons[0]=bnldev(fraction[1]+fraction[2],
			   (int)(nE[m]),&idumi);
      /* nelectrons[1] is the number of electrons initially in depleted rgn*/
      nelectrons[1]=bnldev(fraction[1]/(fraction[1]+fraction[2]),
			   (int)(nelectrons[0]),&idumi); 
      /*nelectrons[2] is the number of electrons initially in field-free rgn*/
      nelectrons[2]=nelectrons[0]-nelectrons[1];

      fprintf(stderr,"REM nelectrons[0,1,2] = %g %g %g fract[1+2] %g fract[1/(1+2)] %g\n",nelectrons[0],nelectrons[1],nelectrons[2],fraction[1]+fraction[2],fraction[1]/(fraction[1]+fraction[2]));

      // NOTE: NEED TO CHANGE 3.34

      /* accumulate the charges from the various parts of the device. */
      for (part=1;part<3;part++) {
	if (nelectrons[part]==0.0) continue;
	if (fraction_moment[part]<depletiondepth-sqrt(0.5)*minDelta) {
	  if (fraction_moment[part]<=0.0) fraction_moment[part]=0.001;
	  // don't compute using ccd_runpars structure..
	  
	  //	  drift_sigma=3.34117*drift_sigma_scale*
	  //	    sqrt(-log(1.-(fraction_moment[part]/(depletiondepth+z_twiddle)))
	  //		 *(temperature/200.0)*(2e12/NA));
	  { // look up drift_sigma here using the sigma(z) array 
	    // from ccd_rp->ccdpars.lateral_sigma
	    float t=(nsigma-1)*
	      (depletiondepth-fraction_moment[part])/depletiondepth;
	    int idx=floor(t);
	    t -= idx;
	    if (idx==nsigma-1) {
	      drift_sigma = (1+t)*lateral_sigma[idx]-t*lateral_sigma[idx-1];
	    } else {
	      drift_sigma = t*lateral_sigma[idx+1]+(1-t)*lateral_sigma[idx];
	    }
	  }
	  sigma=sqrt(pow(init_sigma,2.0)+pow(drift_sigma,2.0));
	  // previous expressions caused a bug, evidently.
	  //	  lamda = PIXEL/(4.*sqrt(sigma));
	  //	  zeta  = 1./(2.*sqrt(sigma));
	  lamda = PIXEL/(2*sqrt(2)*sigma);
	  zeta  = 1./(sqrt(2)*sigma);
	  /* now that charge splitting sigma is calculated, can ammend the
	     true ammount of charge due to boundary crossing */
	  /* y part for charge cloud relocation. */
	  
	  ty = 1. - 2.*fabs(fmod(fabs(y[m]/PIXEL+0.5),1.)-0.5);
	  typ = wallthicknss(fraction_moment[part],ckwalldepth)/(1.0*PIXEL);
	  if (typ==0.0 && fraction_moment[part]<ckwalldepth[1]) {
	    /* work in an effective attraction to the pixel boundaries */
	    typ=4.*ckdeff*
	      ckwalldepth[1]/depletiondepth*
		fabs((fraction_moment[part]-ckwalldepth[1])*
		     fraction_moment[part])/(ckwalldepth[1]*ckwalldepth[1]);
	    last=0.5*PIXEL*((1.-deflection((1.-ty),typ))-ty);
	  } else {
	    last=0.5*PIXEL*(deflection(ty,typ)-ty);
	  }
	  this=2*y[m]/PIXEL;
	  if (((((int)(floor(fabs(this))))%2 == 1) && (this < 0.0)) ||
	      ((((int)(floor(fabs(this))))%2 == 0) && (this >= 0.0))) 
	    { usey=y[m]-last; } else { usey=y[m]+last; }
	  
	  /* x part for charge cloud relocation. */
	  
	  tx = 1. - 2.*fabs(fmod(fabs(x[m]/PIXEL+0.5),1.)-0.5);
	  typ = wallthicknss(fraction_moment[part],cswalldepth)/(1.0*PIXEL);
	  last=0.5*PIXEL*(deflection(tx,typ)-tx);
	  this=2*x[m]/PIXEL;
	  if (((((int)(floor(fabs(this))))%2 == 1) && (this < 0.0)) ||
	      ((((int)(floor(fabs(this))))%2 == 0) && (this >= 0.0))) 
	    { usex=x[m]-last; } else { usex=x[m]+last; }
	  for (j=0;j<=yside;j++) 
	    fracy[j]=erf(-lamda - zeta * (usey - (j+starty)*PIXEL)); 
	  for (j=0;j<yside;j++) 
	    fracy[j]=fracy[j+1]-fracy[j];
	  for (k=0;k<=xside;k++) 
	    fracx[k]=erf(-lamda - zeta * (usex - (k+startx)*PIXEL));
	  for (k=0;k<xside;k++) 
	    fracx[k]=fracx[k+1]-fracx[k];
	  
	  for (j=0;j<yside;j++) {
	    for (k=0;k<xside;k++) {
	      /* now combine the 1-D fractional charges. */
	      map[j*xside+k] = map[j*xside+k] 
		+ nelectrons[part] * 0.25  * fracx[k] * fracy[j];
	    }
	  }
	  fprintf(stderr,"REM here 2..\n");
	} else {
	  fprintf(stderr,"REM here 3..CRASHDUMP BELOW!! \n");
	  float zwhich,rad_dist[NRADS],t,ranval,radius,phi,dft_x,dft_y,zf;
	  float z_depl_drift;
	  int i1,i2,index,xp,yp;

	  /* make up the appropriate ad-mixture to serve as the distribution */

	  if (fullydepleted) {
	    float zeta1,zeta2,zeta,zetaprime;
	    /* map the zwhich according to the cumulative distribution
	       derived for the potential shape between zeta1 & zeta2 */
	    //	    fprintf(stderr,"hello! fulldep\n");
	    zeta1=fdff[0]/zm;
	    zeta2=fdff[1]/zm;
	    zeta=fabs((depletiondepth-fraction_moment[part])/zm);
	    fprintf(stderr,"REM zeta %g zeta1 %g zeta2 %g\n",zeta,zeta1,zeta2);
	    // crash was occuring when zeta<0 .. should zeta be < 0 ??
	    zetaprime=zeta1+
	      (zeta2-zeta1)*(log(zeta/zeta1)+0.5*(zeta*zeta-zeta1*zeta1))/
		(log(zeta2/zeta1)+0.5*(zeta2*zeta2-zeta1*zeta1));
	    fraction_moment[part]=depletiondepth-zm*zetaprime;
	    zwhich=fraction_moment[part]-(depletiondepth-fdff[1]);
	    zf=fraction_moment[part];
	    fprintf(stderr,"REM zm %g, zetaprime %g\n",zm,zetaprime);
	  } else {
	    //	    fprintf(stderr,"hello! not full dep\n");
	    zwhich=fraction_moment[part]-depletiondepth;
	    zf=0.5*(fraction_moment[part]+depletiondepth+z_twiddle
		    -sqrt((fraction_moment[part]-depletiondepth-z_twiddle)*
			  (fraction_moment[part]-depletiondepth-z_twiddle)
			  +4*minDelta*minDelta));
	  }

	  fprintf(stderr,"REM here 4\n");
	  z_depl_drift=fraction_moment[part]-zf;

	  /* the following is a compromise for sharing of the diffusion
	     between the field free region and the drift region. */

	  //	  drift_sigma=3.34117*drift_sigma_scale*
	  //	    sqrt(-log(1.-(zf/(depletiondepth+z_twiddle)))
	  //				   *(temperature/200.0)*(2e12/NA));

	  { // look up drift_sigma here using the sigma(z) array 
	    // from ccd_rp->ccdpars.lateral_sigma
	    float t=(nsigma-1)*
	      (depletiondepth-fraction_moment[part])/depletiondepth;
	    int idx=floor(t);
	    t -= idx;
	    fprintf(stderr,"REM idx %d nsigma %d t %g fractionmoment %g part %d\n",idx,nsigma,t,fraction_moment[part],part);
	    if (idx==nsigma-1) {
	      drift_sigma = (1+t)*lateral_sigma[idx]-t*lateral_sigma[idx-1];
	    } else {
	      drift_sigma = t*lateral_sigma[idx+1]+(1-t)*lateral_sigma[idx];
	    }
	    fprintf(stderr,"REM after if statement..\n");
	  }

	  fprintf(stderr,"REM here 5\n");
	  //	  {
	    //	    int ix=floor((zf/depletiondepth)*ccd_rp->ccdpars.n_sigma);
	    //	    ix=ccd_rp->ccdpars.n_sigma;
	    //	    float t=(zf/depletiondepth)*ccd_rp->ccdpars.n_sigma-ix;
	    //	    float u=1-t;
	    //	    fprintf(stderr,"zf = %f depl = %f t=%f max = %f min = %f\n",
	    //		    zf,depletiondepth,t,
	    //		    ccd_rp->ccdpars.lateral_sigma[0],
	    //		    ccd_rp->ccdpars.lateral_sigma[ccd_rp->ccdpars.n_sigma-1]);
	    //	    drift_sigma=driftsigma(zf,ccd_rp);
	  //	  }

	  sigma=sqrt(pow(init_sigma,2.0)+pow(drift_sigma,2.0)+
		     pow(z_depl_drift,2.0));

	  i1=0;
	  while(i1 < NDISTS-1 && zwhich > zlist[i1]) i1++;
	  if (i1==0) {
	    for(i2=0;i2<NRADS;i2++) {
	      rad_dist[i2]=cdists[i1][i2];
	    }
	  } else {
	    t=(zlist[i1]-zwhich)/(float)(zlist[i1]-zlist[i1-1]);
	    for(i2=0;i2<NRADS;i2++) {
	      rad_dist[i2] = t*cdists[i1-1][i2]+(1.0-t)*cdists[i1][i2];
	    }
	  }

	  /* now the distribution is ready. */
	  for (i1=0;i1<nelectrons[part];i1++) {
	    if ((ranval=ran2(&iduml))>rad_dist[NRADS-1]) continue;
	    /* figure out a good starting index */
	    index=(int)(ranval/rad_dist[NRADS-1]*(NRADS-1));
	    index=(index<0)?0:((index>=NRADS)?NRADS:index);
	    if (ranval<rad_dist[index]) {
	      /* then decrease indices */
	      while (ranval<rad_dist[index] && index>=0) index--;
	      index++;
	    } else {
	      /* then increase indices */
	      while (ranval>rad_dist[index] && index<NRADS-1) index++;
	    }
	    /* now rad_dist[index] > ran_val */
	    t=(rad_dist[index]-ranval)/(rad_dist[index]-rad_dist[index-1]);
	    radius=(1.0-t)*rads[index]+t*rads[index-1];
	    phi=6.28319*ran2(&iduml);
	    dft_x=sigma*gasdev(&idumi);
	    dft_y=sigma*gasdev(&idumi);
	    xp=(int)floor((x[m]+radius*cos(phi)+dft_x+PIXEL/2.0)/PIXEL)-startx;
	    yp=(int)floor((y[m]+radius*sin(phi)+dft_y+PIXEL/2.0)/PIXEL)-starty;
	    if ( xp<0 || xp>=xside || yp<0 || yp>=yside ) continue;
	    map[yp*xside+xp]+=1.0;
	  }
	}
      }
      fprintf(stderr,"REM survive? CRASHDUMP BUG ABOVE!\n");
    }

    for (j=0;j<yside;j++){
      elost=poidev(eloss,&idumi);
      if (elost>=map[j*xside+0]) 
	map[j*xside+0]=0.0;
      else
	map[j*xside+0]=map[j*xside+0] - elost;
      for (k=1;k<xside;k++) {
	elost=poidev(eloss,&idumi);
	if (elost>=map[j*xside+k]) 
	  map[j*xside+k]=0.0 + echo * map[j*xside+k-1];
	else
	  map[j*xside+k]=map[j*xside+k]-elost+echo*map[j*xside+k-1];
      }
    }
    /* now add readout noise, dark frame error and convert to ADU */

    if (!ACCUMULATE_N_FLUSH) {
      for (j=0;j<yside;j++) {
	for (k=0;k<xside;k++) {
	  map[j*xside+k]=(int)
	    floor((map[j*xside+k]+dfe+ron*gasdev(&idumi)) * 3.65 / gain+0.5) ; 
	}
      }
    } else {
      // copy the map onto the frame with the right format & offsets
      // these are really approximate. some information may have been destroyed
      // above
      if (bigmap==NULL) {
	if ((bigmap=(float*)malloc(NX*NY*sizeof(float)))==NULL) {
	  fprintf(stderr,"can't allocate for bigmap??");
	  acis_pix_dist_complain();
	}
	int ii=NX*NY;
	while (ii--) bigmap[ii]=0.0;
	// bigmap is ready to go.
      }
      int X0=(x[0]/PIXEL+0.5)-2*rbn;
      int Y0=(y[0]/PIXEL+0.5)-2*rbn;
      for (j=0;j<yside;j++) {
	for (k=0;k<xside;k++) {
	  if ((Y0+j<0) ||
	      ((NY-1)-(Y0+j)<0) ||
	      (X0+k<0) ||
	      ((NX-1)-(X0+k)<0)) {
	    // out of bounds. do nothing.
	  } else {
	    // increment charge content
	    bigmap[(Y0+j)*NX+(X0+k)] += map[j*xside+k];
	  }
	}
      }
    }

    {
      // added this to do rebinning in place since pixel=10um, depl=100um
      // ends up with events spanning many pixels.
      // when outputting rv data,
      rebin_map(map,rbn,&xside,&yside);
    }

    nout=0;
    switch (operation) {
    case PUTOUT_MAPS:
      for (m=0;m<numphot;m++) 
	printf("# x: %f y: %f z: %f usex: %f usey: %f sigma: %f Fdepleted: %f Ffield-free: %f Mdepleted: %f Mfield-free: %f\n",
	       x[m],y[m],z[m],usex,usey,sigma,fraction[1],fraction[2],fraction_moment[1],fraction_moment[2]);
      dump_local_map(xside,yside,map);
      nout++;
      break;
    case PUTOUT_RV:
      /* now find local maxima */
      /* prepare to place the event where it landed in the pixel. */
      /* y is reversed because of the location of clock #3. */
      if (do_rays) {
	temp1=(int)( floor( x[0]/PIXEL+0.5) );
	temp2=(int)( floor(-y[0]/PIXEL+0.5) );
      } else {
	temp1=(int)( ( x[0]/PIXEL+0.5) * 512);
	temp2=(int)( (-y[0]/PIXEL+0.5) * 512);
      }
      //      fprintf(stderr,"xside,yside = %d,%d\n",xside,yside);
      if (!ACCUMULATE_N_FLUSH) {
	output_evlist(map,xside,yside,outf,&nout,temp1,temp2);
      }
      break;
    default:
      fprintf(stderr,"wrong mode? operation = %d.\n",operation);
      exit(0);
      break;
    }
/*    if (nout>0) i++; */
    if (nout==1 && fluoresce==1) {nfluor++;}
    if (nout==2 && escape==1) {nescape++;nfluor++;}
    nreal += nout;
    nattempt++;
    static int lasti=0;
    if (ACCUMULATE_N_FLUSH && (i-lasti)-ACCUMULATE_N_FLUSH>=0) {
      int ix;
      lasti=i;
      ix=NX*NY;
      while (ix--) 
	bigmap[ix]=
	  (int)floor((bigmap[ix]+dfe+ron*gasdev(&idumi))*3.65/gain+0.5) ; 
      int xoffset=0;
      int yoffset=0;
      int nx=NX;
      int ny=NY;
      rebin_map(bigmap,rbn,&nx,&ny);
      output_evlist(bigmap,nx,ny,outf,&nout,xoffset,yoffset);
	// and wipe out the map.
      ix=NX*NY;
      while (ix--) bigmap[ix]=0;
    }
  }
 finished_rdng:

  if (ACCUMULATE_N_FLUSH) {
    int ix;
    ix=NX*NY;
    while (ix--) 
      bigmap[ix]=
	(int)floor((bigmap[ix]+dfe+ron*gasdev(&idumi))*3.65/gain+0.5) ; 
    int xoffset=0;
    int yoffset=0;
    int nx=NX;
    int ny=NY;
    rebin_map(bigmap,rbn,&nx,&ny);
    output_evlist(bigmap,nx,ny,outf,&nout,xoffset,yoffset);
    // and wipe out the map.
    ix=NX*NY;
    free(bigmap);
  }
  // final call to output
  if ((fp=fopen("cpix_out","w"))!=NULL){
    fprintf(fp,
	    "energy interacting_photons total_photons total_events expected_events escape fluorescence\n");
    fprintf(fp,"%f %d %ld %ld %ld %ld %ld\n",primaryenergy,n,ntries,nreal,nattempt,nescape,nfluor);
    fclose(fp);
  }
  exit(0);
}

void output_evlist (float *map,int xside,int yside,FILE *of,int *nout,int xoff,int yoff) {
  int j,k;
  float *cenpix;
  for (j=1;j<yside-1;j++){
    for (k=1;k<xside-1;k++){
      if (map[j*xside+k] < threshold) continue;
      cenpix=&(map[j*xside+k]);
      if ((*cenpix > *(cenpix-xside-1)) 
	  && (*cenpix > *(cenpix-xside))
	  && (*cenpix > *(cenpix-xside+1))
	  && (*cenpix > *(cenpix-1))
	  && (*cenpix >= *(cenpix+1))
	  && (*cenpix >= *(cenpix+xside-1))
	  && (*cenpix >= *(cenpix+xside))
	  && (*cenpix >= *(cenpix+xside+1)))
	{ /* local maximum hit! */
	  /* again, y is reversed because of the location of clock #3. */
	  event.data[0] = (int) *(cenpix+xside-1);
	  event.data[1] = (int) *(cenpix+xside  );
	  event.data[2] = (int) *(cenpix+xside+1);
	  event.data[3] = (int) *(cenpix      -1);
	  event.data[4] = (int) *(cenpix        );
	  event.data[5] = (int) *(cenpix      +1);
	  event.data[6] = (int) *(cenpix-xside-1);
	  event.data[7] = (int) *(cenpix-xside  );
	  event.data[8] = (int) *(cenpix-xside+1);
	  event.x=(short)(xoff+k);
	  event.y=(short)(yoff+j);
	  fwrite(&event,size,1,of);
	  (*nout)++;
	}
    }
  }
}

float wallthicknss (zdepth,wallstreet)
float zdepth,wallstreet[];
{
  if (wallstreet[0]==0.0) return (0.0);
  if (zdepth<=wallstreet[1] || zdepth>=wallstreet[3]) return (0.0);
  if (zdepth<wallstreet[2]) {
    return(wallstreet[0]*sqrt(1.-pow((zdepth-wallstreet[2])/(wallstreet[1]-wallstreet[2]),2.0)));
/*    return(wallstreet[0]*(zdepth-wallstreet[1])/(wallstreet[2]-wallstreet[1]));*/
  } else {
    return(wallstreet[0]*sqrt(1.-pow((zdepth-wallstreet[2])/(wallstreet[3]-wallstreet[2]),2.0)));
/*    return(wallstreet[0]*(wallstreet[3]-zdepth)/(wallstreet[3]-wallstreet[2]));*/
  }
}

float deflection (ty,typ)
float ty,typ;
{
 return((1.-exp(-ty*(exp(2*typ)-0.99999)))/(1.-exp(-(exp(2*typ)-0.99999))));
}

float atten_coeff ( energy, pars)
     float energy,pars[];
     /* routine to calculate the attenuation coefficient    */
     /* for a given energy and the given set of parameters. */ 
     /* based on KCG's fits to HENKE data.                  */
{
  float temp;
  temp=pars[0]*pow(energy,pars[1])*(1.+pars[2]*pow(energy,pars[3]));
  return (1./temp);
}

void get_optical_depths(float *xpos,float t[],float ts[],float imfps[],
			float imfpsks[],float p_si[],float epel[])
//     float *xpos,t[],ts[],imfps[],imfpsks[],p_si[],epel[];
{
  if ( *xpos < WCS ) 
    {
      t[0]=TCSP*imfp_si;
      t[1]=TCSO*imfp_sio2;
      t[2]=si1*imfp_si;
      t[3]=so1*imfp_sio2;
      t[4]=si2*imfp_si;
      t[5]=so2*imfp_sio2;
      t[6]=0.0;
      ts[0]=TCSP*imfpsk_si;
      ts[1]=TCSO*imfpsk_sio2;
      ts[2]=si1*imfpsk_si;
      ts[3]=so1*imfpsk_sio2;
      ts[4]=si2*imfpsk_si;
      ts[5]=so2*imfpsk_sio2;
      ts[6]=0.0;
      imfps[0]=imfp_si;
      imfps[1]=imfp_sio2;
      imfps[2]=imfp_si;
      imfps[3]=imfp_sio2;
      imfps[4]=imfp_si;
      imfps[5]=imfp_sio2;
      imfps[6]=1;
      imfpsks[0]=imfpsk_si;
      imfpsks[1]=imfpsk_sio2;
      imfpsks[2]=imfpsk_si;
      imfpsks[3]=imfpsk_sio2;
      imfpsks[4]=imfpsk_si;
      imfpsks[5]=imfpsk_sio2;
      imfpsks[6]=1;
      p_si[0]=1.0;
      p_si[1]=28.0/60.0*imfp_si/imfp_sio2;
      p_si[2]=1.0;
      p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
      p_si[4]=1.0;
      p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
      p_si[6]=0.0;
      epel[0]=0.00365;
      epel[1]=0.00365*9.0/1.1;
      epel[2]=0.00365;
      epel[3]=0.00365*9.0/1.1;
      epel[4]=0.00365;
      epel[5]=0.00365*9.0/1.1;
      epel[6]=0.00365;
    }
  else 
    if (*xpos < WCS+2*TCSP) {
      t[0]=(TCSP+(TCSO-TIO)/2.)*imfp_si;
      t[1]=TIO*imfp_sio2;
      t[2]=TIN*imfp_si3n4;
      t[3]=((TCSO-TIO)/2.-TIN+si1)*imfp_si;
      t[4]=so1*imfp_sio2;
      t[5]=si2*imfp_si;
      t[6]=so2*imfp_sio2;
      ts[0]=(TCSP+(TCSO-TIO)/2.)*imfpsk_si;
      ts[1]=TIO*imfpsk_sio2;
      ts[2]=TIN*imfpsk_si3n4;
      ts[3]=((TCSO-TIO)/2.-TIN+si1)*imfpsk_si;
      ts[4]=so1*imfpsk_sio2;
      ts[5]=si2*imfpsk_si;
      ts[6]=so2*imfpsk_sio2;
      imfps[0]=imfp_si;
      imfps[1]=imfp_sio2;
      imfps[2]=imfp_si3n4;
      imfps[3]=imfp_si;
      imfps[4]=imfp_sio2;
      imfps[5]=imfp_si;
      imfps[6]=imfp_sio2;
      imfpsks[0]=imfpsk_si;
      imfpsks[1]=imfpsk_sio2;
      imfpsks[2]=imfpsk_si3n4;
      imfpsks[3]=imfpsk_si;
      imfpsks[4]=imfpsk_sio2;
      imfpsks[5]=imfpsk_si;
      imfpsks[6]=imfpsk_sio2;
      p_si[0]=1.0;
      p_si[1]=28.0/60.0*imfp_si/imfp_sio2;
      p_si[2]=84.0/140.0*imfp_si/imfp_si3n4;
      p_si[3]=1.0;
      p_si[4]=28.0/60.0*imfp_si/imfp_sio2;
      p_si[5]=1.0;
      p_si[6]=28.0/60.0*imfp_si/imfp_sio2;
      epel[0]=0.00365;
      epel[1]=0.00365*9.0/1.1;
      epel[2]=0.00365*5.0/1.1;
      epel[3]=0.00365;
      epel[4]=0.00365*9.0/1.1;
      epel[5]=0.00365;
      epel[6]=0.00365*9.0/1.1;
    }
    else 
      if (*xpos < WCS+2*(TCSP+si1)) {
	t[0]=TIO*imfp_sio2;
	t[1]=TIN*imfp_si3n4;
	t[2]=((TCSO-TIO)/2.-TIN+si1)*imfp_si;
	t[3]=so1*imfp_sio2;
	t[4]=si2*imfp_si;
	t[5]=so2*imfp_sio2;
	t[6]=0.0;
	ts[0]=TIO*imfpsk_sio2;
	ts[1]=TIN*imfpsk_si3n4;
	ts[2]=((TCSO-TIO)/2.-TIN+si1)*imfpsk_si;
	ts[3]=so1*imfpsk_sio2;
	ts[4]=si2*imfpsk_si;
	ts[5]=so2*imfpsk_sio2;
	ts[6]=0.0;
	imfps[0]=imfp_sio2;
	imfps[1]=imfp_si3n4;
	imfps[2]=imfp_si;
	imfps[3]=imfp_sio2;
	imfps[4]=imfp_si;
	imfps[5]=imfp_sio2;
	imfps[6]=1;
	imfpsks[0]=imfpsk_sio2;
	imfpsks[1]=imfpsk_si3n4;
	imfpsks[2]=imfpsk_si;
	imfpsks[3]=imfpsk_sio2;
	imfpsks[4]=imfpsk_si;
	imfpsks[5]=imfpsk_sio2;
	imfpsks[6]=1;
	p_si[0]=28.0/60.0*imfp_si/imfp_sio2;
	p_si[1]=84.0/140.0*imfp_si/imfp_si3n4;
	p_si[2]=1.0;
	p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
	p_si[4]=1.0;
	p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
	p_si[6]=0.0;
	epel[0]=0.00365*9.0/1.1;
	epel[1]=0.00365*5.0/1.1;
	epel[2]=0.00365;
	epel[3]=0.00365*9.0/1.1;
	epel[4]=0.00365;
	epel[5]=0.00365*9.0/1.1;
	epel[6]=0.00365;
      }
      else
	if (*xpos < WCS+2*(TCSP+si1+so1)) {
	  t[0]=TIO*imfp_sio2;
	  t[1]=TIN*imfp_si3n4;
	  t[2]=si1*imfp_si;
	  t[3]=((TCSO-TIO)/2.-TIN+so1)*imfp_sio2;
	  t[4]=si2*imfp_si;
	  t[5]=so2*imfp_sio2;
	  t[6]=0.0;
	  ts[0]=TIO*imfpsk_sio2;
	  ts[1]=TIN*imfpsk_si3n4;
	  ts[2]=si1*imfpsk_si;
	  ts[3]=((TCSO-TIO)/2.-TIN+so1)*imfpsk_sio2;
	  ts[4]=si2*imfpsk_si;
	  ts[5]=so2*imfpsk_sio2;
	  ts[6]=0.0;
	  imfps[0]=imfp_sio2;
	  imfps[1]=imfp_si3n4;
	  imfps[2]=imfp_si;
	  imfps[3]=imfp_sio2;
	  imfps[4]=imfp_si;
	  imfps[5]=imfp_sio2;
	  imfps[6]=1;
	  imfpsks[0]=imfpsk_sio2;
	  imfpsks[1]=imfpsk_si3n4;
	  imfpsks[2]=imfpsk_si;
	  imfpsks[3]=imfpsk_sio2;
	  imfpsks[4]=imfpsk_si;
	  imfpsks[5]=imfpsk_sio2;
	  imfpsks[6]=1;
	  p_si[0]=28.0/60.0*imfp_si/imfp_sio2;
	  p_si[1]=84.0/140.0*imfp_si/imfp_si3n4;
	  p_si[2]=1.0;
	  p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
	  p_si[4]=1.0;
	  p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
	  p_si[6]=0.0;
	  epel[0]=0.00365*9.0/1.1;
	  epel[1]=0.00365*5.0/1.1;
	  epel[2]=0.00365;
	  epel[3]=0.00365*9.0/1.1;
	  epel[4]=0.00365;
	  epel[5]=0.00365*9.0/1.1;
	  epel[6]=0.00365;
	}
	else
	  if (*xpos < WCS+2*(TCSP+si1+so1+si2)) {
	    t[0]=TIO*imfp_sio2;
	    t[1]=TIN*imfp_si3n4;
	    t[2]=si1*imfp_si;
	    t[3]=so1*imfp_sio2;
	    t[4]=((TCSO-TIO)/2.-TIN+si2)*imfp_si;
	    t[5]=so2*imfp_sio2;
	    t[6]=0.0;
	    ts[0]=TIO*imfpsk_sio2;
	    ts[1]=TIN*imfpsk_si3n4;
	    ts[2]=si1*imfpsk_si;
	    ts[3]=so1*imfpsk_sio2;
	    ts[4]=((TCSO-TIO)/2.-TIN+si2)*imfpsk_si;
	    ts[5]=so2*imfpsk_sio2;
	    ts[6]=0.0;
	    imfps[0]=imfp_sio2;
	    imfps[1]=imfp_si3n4;
	    imfps[2]=imfp_si;
	    imfps[3]=imfp_sio2;
	    imfps[4]=imfp_si;
	    imfps[5]=imfp_sio2;
	    imfps[6]=1;
	    imfpsks[0]=imfpsk_sio2;
	    imfpsks[1]=imfpsk_si3n4;
	    imfpsks[2]=imfpsk_si;
	    imfpsks[3]=imfpsk_sio2;
	    imfpsks[4]=imfpsk_si;
	    imfpsks[5]=imfpsk_sio2;
	    imfpsks[6]=1;
	    p_si[0]=28.0/60.0*imfp_si/imfp_sio2;
	    p_si[1]=84.0/140.0*imfp_si/imfp_si3n4;
	    p_si[2]=1.0;
	    p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
	    p_si[4]=1.0;
	    p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
	    p_si[6]=0.0;
	    epel[0]=0.00365*9.0/1.1;
	    epel[1]=0.00365*5.0/1.1;
	    epel[2]=0.00365;
	    epel[3]=0.00365*9.0/1.1;
	    epel[4]=0.00365;
	    epel[5]=0.00365*9.0/1.1;
	    epel[6]=0.00365;
	  }
	  else 
	    if (*xpos < WCS+2*(TCSP+si1+so1+si2+so2)) {
	      t[0]=TIO*imfp_sio2;
	      t[1]=TIN*imfp_si3n4;
	      t[2]=si1*imfp_si;
	      t[3]=so1*imfp_sio2;
	      t[4]=si2*imfp_si;
	      t[5]=((TCSO-TIO)/2.-TIN+so2)*imfp_sio2;
	      t[6]=0.0;
	      ts[0]=TIO*imfpsk_sio2;
	      ts[1]=TIN*imfpsk_si3n4;
	      ts[2]=si1*imfpsk_si;
	      ts[3]=so1*imfpsk_sio2;
	      ts[4]=si2*imfpsk_si;
	      ts[5]=((TCSO-TIO)/2.-TIN+so2)*imfpsk_sio2;
	      ts[6]=0.0;
	      imfps[0]=imfp_sio2;
	      imfps[1]=imfp_si3n4;
	      imfps[2]=imfp_si;
	      imfps[3]=imfp_sio2;
	      imfps[4]=imfp_si;
	      imfps[5]=imfp_sio2;
	      imfps[6]=1;
	      imfpsks[0]=imfpsk_sio2;
	      imfpsks[1]=imfpsk_si3n4;
	      imfpsks[2]=imfpsk_si;
	      imfpsks[3]=imfpsk_sio2;
	      imfpsks[4]=imfpsk_si;
	      imfpsks[5]=imfpsk_sio2;
	      imfpsks[6]=1;
	      p_si[0]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[1]=84.0/140.0*imfp_si/imfp_si3n4;
	      p_si[2]=1.0;
	      p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[4]=1.0;
	      p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[6]=0.0;
	      epel[0]=0.00365*9.0/1.1;
	      epel[1]=0.00365*5.0/1.1;
	      epel[2]=0.00365;
	      epel[3]=0.00365*9.0/1.1;
	      epel[4]=0.00365;
	      epel[5]=0.00365*9.0/1.1;
	      epel[6]=0.00365;
	    }
	    else {
	      t[0]=TIO*imfp_sio2;
	      t[1]=TIN*imfp_si3n4;
	      t[2]=si1*imfp_si;
	      t[3]=so1*imfp_sio2;
	      t[4]=si2*imfp_si;
	      t[5]=so2*imfp_sio2;
	      t[6]=0.0;
	      ts[0]=TIO*imfpsk_sio2;
	      ts[1]=TIN*imfpsk_si3n4;
	      ts[2]=si1*imfpsk_si;
	      ts[3]=so1*imfpsk_sio2;
	      ts[4]=si2*imfpsk_si;
	      ts[5]=so2*imfpsk_sio2;
	      ts[6]=0.0;
	      imfps[0]=imfp_sio2;
	      imfps[1]=imfp_si3n4;
	      imfps[2]=imfp_si;
	      imfps[3]=imfp_sio2;
	      imfps[4]=imfp_si;
	      imfps[5]=imfp_sio2;
	      imfps[6]=1;
	      imfpsks[0]=imfpsk_sio2;
	      imfpsks[1]=imfpsk_si3n4;
	      imfpsks[2]=imfpsk_si;
	      imfpsks[3]=imfpsk_sio2;
	      imfpsks[4]=imfpsk_si;
	      imfpsks[5]=imfpsk_sio2;
	      imfpsks[6]=1;
	      p_si[0]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[1]=84.0/140.0*imfp_si/imfp_si3n4;
	      p_si[2]=1.0;
	      p_si[3]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[4]=1.0;
	      p_si[5]=28.0/60.0*imfp_si/imfp_sio2;
	      p_si[6]=0.0;
	      epel[0]=0.00365*9.0/1.1;
	      epel[1]=0.00365*5.0/1.1;
	      epel[2]=0.00365;
	      epel[3]=0.00365*9.0/1.1;
	      epel[4]=0.00365;
	      epel[5]=0.00365*9.0/1.1;
	      epel[6]=0.00365;
	    }
}

void makeODmaps (int dim)
{
  int i,j,k;
  float x0,y0,tau[7],tausk[7],imfps[7],imfpsks[7],t;
  float likelihood_Si[7],en_elec[7];

  for (j=0;j<dim;j++){
    y0=(j/(float)(dim-1))*PIXEL-PIXEL/2.;
    y0=PIXEL/2.+y0;
    k=0;
    /*    printf("%d\n",dim);*/
    while ( y0 > yBND[k] ) k++;
    si1=SI1[k]; 
    so1=SO1[k];
    si2=SI2[k];
    so2=SO2[k];
    for (i=0;i<dim;i++){
      x0=(i/(float)(dim-1))*PIXEL-PIXEL/2.;
      x0=PIXEL-fabs(2*x0);
      get_optical_depths(&x0,tau,tausk,imfps,imfpsks,likelihood_Si,en_elec);   
      t=0.0;
      for (k=0;k<7;k++) t=t+tau[k];
      fprintf (outf,"%f ",t);
    }
    fprintf(outf,"\n");
  }
}

void mk_bs_cumdist(int nrads,float rads[],float maxrad,float cdist[],
		   float z,float bs,float R,float L,int nalpha,float alpha[])
//int nrads,nalpha;
//float rads[],maxrad,cdist[],z,bs,R,L,alpha[];
{
  int loop,i,j,k;
  float T,a1,a2,beta,fraction,incr,psi;

  T=1.0-R;a1=bs/L;a2=(bs-z)/L;beta=bs*T/(L*R);
  fraction=(R*cosh(a2)+T*sinh(a2))/(R*cosh(a1)+T*sinh(a1));
  rads[0]=0.0;
  cdist[0]=0.0;
  for(i=1;i<nrads;i++){
    rads[i]=MINRAD*exp((i-1)*log(maxrad/MINRAD)/(float)(nrads-1.0));
    cdist[i]=fraction;
    loop=1;
    psi=sqrt(pow(alpha[loop],2.0)+pow(bs/L,2.0));
    while ((incr=bessk1(rads[i]/bs*psi)) > 1.e-8 && loop<nalpha) {
      incr *= 2*alpha[loop]*beta*
	sin(alpha[loop]*z/bs)/
	  (beta+pow(cos(alpha[loop]),2.0))*rads[i]/(psi*bs);
      cdist[i]-=incr;
      loop++;
      psi=sqrt(pow(alpha[loop],2.0)+pow(bs/L,2.0));
    }
  }
}

void mk_alphas(int nalpha,float alpha[],float bs,float L,float R)
//int nalpha;
//float alpha[],bs,L,R;
{
  int i,j;
  float beta;

  beta=bs*(1.0-R)/(L*R);
  for(i=1;i<nalpha;i++) {
    /* use a reasonable starting value */
    alpha[i]=j*PI;
    /* do this iteratively 100 times */
    for(j=0;j<100;j++) alpha[i]=i*PI+atan(-alpha[i]/beta);
  }
}

void setup_bsprofs(int nrads,float rads[],float maxrad,int ndists,
		   float *cdists[],float z[],float bs,float R,float L,
		   int nalpha,float alpha[])
//int nrads,ndists,nalpha;
//float rads[],maxrad,*cdists[],bs,R,L,alpha[],z[];
{
  int i,j;

  mk_alphas(nalpha,alpha,bs,L,R);
  for(i=0;i<ndists;i++){
    z[i]=MINZ*exp(i*log(bs/MINZ)/(float)(ndists-1.0));
    mk_bs_cumdist(nrads,rads,maxrad,cdists[i],z[i],bs,R,L,nalpha,alpha);
  }

/*
printf("%f ",-1.0);
for(i=0;i<ndists;i++){
printf("%f ",z[i]);
}
for(i=0;i<nrads;i++){
printf("\n%f ",rads[i]);
for(j=0;j<ndists;j++){
printf("%f ",*(cdists[j]+i));
}
}
*/

}

void setup_mfps(float energy,float cofda[][4])
{
  /* mfp_si == mean free path in silicon; */
  /*  mfp_sio2 == mean free path in sio2. */

  // use HENKE values instead of KCG's fit. Do this only for Si. energy will 
  // remain constant for each invocation.
  // not yet implemented (using KCG's fits)
  if (energy > ESiKEdge) {
    imfp_si   =atten_coeff(energy,cofda[0]);
    imfp_sio2 =atten_coeff(energy,cofda[1]);
    imfp_si3n4=atten_coeff(energy,cofda[2]);
  } else {
    if (energy > EOKEdge) {
      imfp_si   =atten_coeff(energy,cofda[3]);
      imfp_sio2 =atten_coeff(energy,cofda[4]);
      imfp_si3n4=atten_coeff(energy,cofda[5]);
    } else {
      if (energy > ENKEdge) {
	imfp_si   =atten_coeff(energy,cofda[6]);
	imfp_sio2 =atten_coeff(energy,cofda[7]);
	imfp_si3n4=atten_coeff(energy,cofda[8]);
      } else {
	imfp_si   =atten_coeff(energy,cofda[9]);
	imfp_sio2 =atten_coeff(energy,cofda[10]);
	imfp_si3n4=atten_coeff(energy,cofda[11]);
      }
    }
  }

  /* imfpsk_si & imfpsk_sio2 etc.. mfp for Si K fluorescence. */
  imfpsk_si   = atten_coeff(ESiK,cofda[3]);
  imfpsk_sio2 = atten_coeff(ESiK,cofda[4]);
  imfpsk_si3n4= atten_coeff(ESiK,cofda[5]);
  // and finally here overwrite imfpsk_si imfp_si using henke numbers.
  imfpsk_si=1e-4*abs_coeff(12.398/ESiK,173.0); // micron^-1
  imfp_si=1e-4*abs_coeff(12.398/energy,173.0); // micron^-1
}

void dump_local_map(int xs,int ys,float *map)
//int   xs,ys;
//float *map;
{
  int i,j;
  for (i=0;i<ys;i++) {
    for (j=0;j<xs;j++) {
      fprintf(outf,"%f ",map[i*xs+j]);
    }
    fprintf(outf,"\n");
  }
  fprintf(outf,"\n");
}

void acis_pix_dist_complain(void)
{
  fprintf (stderr,"%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n",
"usage: cpix_dist -e <energy> (-n <depl_events>| -N <photons>)",
"    [-ne | -r <rms noise (electrons)>][-t <event threshold (ADU)>]",
"    [-g <gain (eV per ADU)>][-o <output filename>][-T <kelvin>][-z <z1><z2>]",
"    [-s <sensor number [0|1]>][-c <chip number [0|1|2|3]>][-k <echo factor>]",
"    [-wc (CS_wall)<w><d1><d2><d3>][-wk (CLK_wall)<w><d1><d2><d3>]",
"    [-d <depl.depth>][-D <resid. dferror (e-)>][-l <electron_loss>]",
"    [-m <diagnostic mode [opticaldepthmap [dim]|eventmaps|rv_file]>]",
"    [-f <clock deflection factor>][-B (backside option)][-b <bs Si>]",
"    [-x (excl. fluorescence & escape features)][-C (excl. fs partial evts)]",
"    [-Y <Si Fluorescent Yield>][-br <backside_reflection_coefficient>]",
"    [-L <bs diff.length>][-F (`fully depleted')][-P <channel potential>]",
"    [-kT <Schottky barrier `ff' thresh>][-M <minimum_delta_scale_factor>]",
"    [-S <drift_sigma_scale>][-R <resonance_collisions>]",
"NOTE: specifying a sensor and a chip will override any gains/noises specified."
);
  exit (1);
}

void rebin_map (float *map,int rbn,int *xs,int *ys) {
  float *newmap=(float*)malloc((*xs/rbn)*(*ys/rbn)*sizeof(float));
  int ix=(*xs/rbn)*(*ys/rbn);
  int jx,kx,extrax,extray;
  
  while (ix--) newmap[ix]=0.0;
  // depending on tmp1%rbn & tmp2%rbn,
  // rebinning boundaries will vary.
  int tmp1=rbn*(rand()/(float)RAND_MAX);
  int tmp2=rbn*(rand()/(float)RAND_MAX);
  extrax=(tmp1+rbn*4096)%rbn;
  extray=(tmp2+rbn*4096)%rbn;
  
  for (jx=0;jx<*xs;jx++) {
    for (kx=0;kx<*ys;kx++) {
      if (((kx+extray)/rbn < *ys/rbn) &&
	  ((jx+extrax)/rbn < *xs/rbn)) {
	newmap[((kx+extray)/rbn)*((*xs)/rbn)+((jx+extrax)/rbn)]
	  +=map[kx*(*xs)+jx];
      }
    }
  }
  *xs /= rbn;
  *ys /= rbn;
  memcpy(map,newmap,(*xs)*(*ys)*sizeof(float));
  free(newmap);
}
