#include <stdio.h>
#include <stdlib.h>
#include "numrec/include/nr.h"
#include "numrec/include/nrutil.h"
#include "henke.h"


extern int  complain (char *s);

void interp_henke (henke_file *h,float energy, float *f1, float *f2) {
  static int index=0;
  float t;
  energy*=1000.0;
  hunt((h->energy)-1,h->n_entries,energy,&index);
  index=(index==0)?0:index-1;
  t=(energy - h->energy[index])/(h->energy[index+1] - h->energy[index]);
  *f1 = t * h->f1[index+1] + (1.0-t) * h->f1[index];
  *f2 = t * h->f2[index+1] + (1.0-t) * h->f2[index];
}

void read_henke_file (henke_file *h) {
  int nline;
  FILE *fp;
  char line[HENKE_STRLEN],err[2048];

  if ((fp=fopen(h->filename,"r"))==NULL) {
    sprintf(err,"can't access %s%s%s%s", h->filename,
	    "\n\nis the environment variable ",henke_dir,
	    " defined correctly?");
    fprintf(stderr,"%s\n",err);
    exit(1);
  }
  nline=-1;
  while(fgets(line,HENKE_STRLEN,fp)) {
    if ((nline>=0) && (nline<MAX_HENKE_ENTRIES)) {
      sscanf(line,"%f %f %f",&h->energy[nline],&h->f1[nline],&h->f2[nline]);
    }
    nline++;
  }
  h->n_entries=(nline>MAX_HENKE_ENTRIES)?MAX_HENKE_ENTRIES:nline;
}

