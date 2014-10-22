#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include "henke.h"
#include "ray.h"
#include "bi_ccd.h"

#define E_RADIUS 2.81794e-13 // cm

char err[2048];

int complain_bi_ccd_Si (char *s);

float n_silicon (float lamda) {
  float energy=12398/lamda;
  // data points were digitized from Philipp & Taft (1960) and fit between 0 and 3.4 eV
  return(3.364967 + 0.2119184*energy + 2.78878*exp((energy-3.30168)/0.397862));
  // some outdated 
  //  return(5.6);
}

float abs_coeff(float lamda,float T) {
  // follows Rajkanan et al. Solid-state electronics 22, pp793-795.
  // uses henke lookup for short wavelength.
  static float last_lamda=-1,last_T=-1;
  static float Eg_T[2],Eg_0[]={1.1557,2.5}; // eV
  static float Egd_T,Egd_0=3.2;          // eV
  static float Ep[]={1.827e-2,5.773e-2}; // eV
  static float C[]={5.5,4.0};            // no dimension
  static float A[]={3.231e2,7.237e3};    // cm-1 eV-2
  static float Ad = 1.052e6;             // cm-1 eV-2
  static float k_boltzmann=8.617e-5;     // eV K-1
  float eV;                              // eV
  float beta = 7.021e-4;                 // eV K-1
  float gamma=1108;                      // K
  int   i,j;
  static float alpha;
  static float delta_e0,delta_e1[2][2][2]; // eV
  static henke_file Si;


  if ((bi_ccd_xray_wavelengths == 1) && 
      (last_lamda == -1)             && 
      (last_T == -1)) {
    // not initialized. setup henke file.
    //    if (getenv(henke_dir)==NULL) {
    if (! getenv(henke_dir)) {
      sprintf(err,"%s%s\n%s\n%s\n%s\n%s\n%s%s%s\n",
	      "\nERROR!!\n\nyou must set the environment variable ",henke_dir,
	      "to point to the directory containing Henke scattering",
	      "form factors.  Specifically, the files \"si.nff\", \"o.nff\","
	      "\"mg.nff\", \"f.nff\" and \"al.nff\" are required",
	      "in order for ccd_effic to work correctly.",
	      "e.g.,",
	      "setenv ",henke_dir," /usr/local/data/");

      complain_bi_ccd_Si(err);
    }

    sprintf(Si.filename,"%s/%s",getenv(henke_dir),"si.nff");
    read_henke_file(&Si);
    fprintf(stderr,"opened and read HENKE file %s\n",Si.filename);
  }
    

  if (T != last_T) {
    Egd_T   = Egd_0   - (beta*T*T/(T+gamma));
    Eg_T[0] = Eg_0[0] - (beta*T*T/(T+gamma));
    Eg_T[1] = Eg_0[1] - (beta*T*T/(T+gamma));
    last_T=T;
    last_lamda=-1;
  }

  if (lamda != last_lamda) {
    //    if (lamda<413) {
    if (bi_ccd_xray_wavelengths && (lamda<1200)) {
      float f1,f2;
      interp_henke(&Si,12.398/lamda,&f1,&f2);
      alpha=(2.32/28.086*6.02e23*2*E_RADIUS*lamda*1e-8*f2); // cm-1
    } else {
      if (lamda>3100) {
	eV=12398.19462/lamda;
	delta_e0=eV-Egd_T;
	for (j=0;j<2;j++) {
	  for (i=0;i<2;i++) {
	    delta_e1[i][j][0]=eV-Eg_T[j]+Ep[i];
	    delta_e1[i][j][1]=eV-Eg_T[j]-Ep[i];
	  }
	}
	
	alpha=Ad*sqrt((delta_e0>0)?delta_e0:0);
	for (i=0;i<2;i++) {
	  for (j=0;j<2;j++) {
	    float f,c[2];
	    c[0]=((delta_e1[i][j][0]>0)?
		  pow(delta_e1[i][j][0],2)/(exp(Ep[i]/(k_boltzmann*T))-1):0);
	    c[1]=((delta_e1[i][j][1]>0)?
		  pow(delta_e1[i][j][1],2)/(1-exp(-Ep[i]/(k_boltzmann*T))):0);
	    f = C[i]*A[j]*(c[0]+c[1]);
	    alpha += f;
	  }
	}
	// alpha is in cm-1.
      } else {
	// wavelength range is between 413 & 3100 Angstroms - not well characterized. use a constant.
	alpha=1.35e6;
      }
    }
    last_lamda=lamda;
  }
  return(alpha);
}

int complain_bi_ccd_Si (char *s) {
  fprintf(stderr,"%s",s);
  exit(1);
}
