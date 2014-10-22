#include <stdio.h>
#ifndef MULTILAYER
#include "multilayer.h"
#define MULTILAYER
#endif

typedef struct complex_n_struct {
  int     inited;
  int     n_entries;
  float   *wave;
  complex *n;
  char    *filename;
} complex_n_struct;

int store_cpx_index_file (char *filename,complex_n_struct *cpxn);
int interp_cpx_index (float lam,complex_n_struct *cpxn,complex *n);
complex_n_struct *init_cpx_index(char *filename);

