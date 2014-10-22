#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multilayer.h"
#include "complex_n_server.h"
#include "numrec/include/nrutil.h"
#include "numrec/include/nr.h"

// program to slurp in and interpolate data and to return a complex
// index of refraction. the input data needs to be 3 columns, with
// col 1: wavelength [nm]
// col 2: Re(n~)
// col 3: Im(n~)

int tmp_main(int argc,char *argv[]) {
  complex_n_struct *this;
  complex n;
  float lambda;

  this=init_cpx_index("test_cpx_n_Au.txt");

  for (lambda=20;lambda<=2500;lambda+=1.0) {
    interp_cpx_index(lambda,this,&n);
    printf("%f %f %f\n",lambda,n.comp[0],n.comp[1]);
  }
  exit(0);
}

complex_n_struct *init_cpx_index(char *filename) {
  complex_n_struct *this;
  this=(complex_n_struct*)malloc(sizeof(complex_n_struct));
  this->filename=filename;
  this->inited=0;
  return(this);
}

int interp_cpx_index (float lam,complex_n_struct *cpxn,complex *n) {
  // interpolate cpxn for the value of lam and store results in *n
  static int lowindex=0;
  if (!cpxn->inited) {
    // on first invocation, read in the file requested..
    store_cpx_index_file(cpxn->filename,cpxn);
    // now should be ready to interpolate..
    cpxn->inited=1;
  }
  hunt(cpxn->wave-1,cpxn->n_entries,lam,&lowindex);
  // do something special if lowindex=0: then will need to
  // interpolate using indices 0 & 1. normally, we interplate using
  // indices lowindex-1 and lowindex..
  if (lowindex>0) lowindex--;
  {
    float u;
    int i;
    complex n_interp;
    u=(lam-cpxn->wave[lowindex])/
      (cpxn->wave[lowindex+1]-cpxn->wave[lowindex]);
    for (i=0;i<2;i++) {
      n_interp.comp[i] = 
	u*cpxn->n[lowindex+1].comp[i] + (1-u)*cpxn->n[lowindex].comp[i];
    }
    memcpy(n,&n_interp,sizeof(complex));
  }
  return(0);
}

int store_cpx_index_file (char *filename,complex_n_struct *cpxn) {
  FILE *fp;
  float a[3];
  int   n_entries;
  if (cpxn->filename==NULL)
    cpxn->filename=filename;
  if ((fp=fopen(cpxn->filename,"r"))==NULL) {
    fprintf(stderr,"can't open file.. exiting.\n");
    exit(1);
  }
  n_entries=0;
  do {
    if (fscanf(fp,"%f %f %f\n",&a[0],&a[1],&a[2])!=3) {
      fprintf(stderr,
	      "should be getting 3 numbers on a line here..\nexiting.\n");
      exit(1);
    } else {
      n_entries++;
    }
  } while (!feof(fp));
  fprintf(stderr,"opened complex n file %s containing %d entries.\n",filename,n_entries);
  cpxn->wave=(float*)malloc(n_entries*sizeof(float));
  cpxn->n=(complex*)malloc(n_entries*sizeof(complex));
  rewind(fp);
  n_entries=0;
  do {
    if (fscanf(fp,"%f %f %f\n",&a[0],&a[1],&a[2])!=3) {
      fprintf(stderr,
	      "should be getting 3 numbers on a line here..\nexiting.\n");
      exit(1);
    } else {
      cpxn->wave[n_entries]=a[0];
      cpxn->n[n_entries].comp[0]=a[1];
      cpxn->n[n_entries].comp[1]=a[2];
      n_entries++;
    }
  } while (!feof(fp));
  cpxn->n_entries=n_entries;
  return(0);
}
